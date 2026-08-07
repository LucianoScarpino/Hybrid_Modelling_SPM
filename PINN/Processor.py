import torch
import warnings

from torch.utils.data import TensorDataset
from datetime import datetime
from pathlib import Path

from NeuralNetwork import MLP,FFN

class Processing(object):
    """Train and diagnose the simplified-physics PINN.

    It combines data, PDE, and surface-boundary losses during Adam training,
    optionally refines with L-BFGS, and stores the selected checkpoint.
    """

    def __init__(
            self,
            d_loader_train,
            f_loader_train,
            b_loader_train,
            d_loader_val,
            f_loader_val,
            b_loader_val,
            device
            ):
        self.d_loader = d_loader_train
        self.f_loader = f_loader_train
        self.b_loader = b_loader_train
        self.d_val_loader = d_loader_val
        self.f_val_loader = f_loader_val
        self.b_val_loader = b_loader_val
        self.device = device

        if not (len(self.d_loader) == len(self.f_loader) == len(self.b_loader)):
            raise ValueError("Training loaders have different numbers of batches.")

        if not (len(self.d_val_loader) == len(self.f_val_loader) == len(self.b_val_loader)):
            raise ValueError("Validation loaders have different numbers of batches.")

    def train(self,
        epochs,
        init_concentration,
        lambda_d,
        lambda_f,
        lambda_b,
        flux,
        lr = 1e-3,
        early_stopping_patience=800,
        sheduler_patience=200,
        lbfgs_max_iter=500
        ):
        """Fit the PINN with weighted data, PDE, and boundary losses.

        Inputs specify optimization settings and physical constants. Returns
        the best trained concentration model.
        """

        model = MLP().to(self.device)
        opt1 = torch.optim.Adam(params=model.parameters(),lr=lr)

        sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=opt1,
            mode="min",
            factor=0.5,
            patience=sheduler_patience,
            threshold=1e-4,
            threshold_mode="rel",
            cooldown=50,
            min_lr=1e-6
        )

        plateau_reached = False
        best_data_val_loss = float("inf")
        best_total_val_loss = float("inf")
        best_model_state = None
        patience = 0


        for epoch in range(epochs):
            model.train()

            data_loss_sum = 0.0
            pde_loss_sum = 0.0
            boundary_loss_sum = 0.0

            number_data_points = 0
            number_collocation_points = 0
            number_boundary_points = 0

            for batch_index,((dx_train, dy_train),(fx_train,),(bx_train,)) in enumerate(zip(self.d_loader,self.f_loader,self.b_loader)):
                opt1.zero_grad()
                dx_train = dx_train.to(self.device)
                dy_train = dy_train.to(self.device)
                fx_train = fx_train.to(self.device)
                bx_train = bx_train.to(self.device)
                
                x_collocation = fx_train.detach().clone().requires_grad_(True)
                x_boundary = bx_train.detach().clone().requires_grad_(True)

                concentrations_d = model(dx_train,init_concentration)                                                             #concentrations
                concentrations_f = model(x_collocation,init_concentration)                                                        #concentrations
                concentrations_b = model(x_boundary,init_concentration)                                                           #concentrations 

                #PDE
                f_first_derivatives = torch.autograd.grad(
                    concentrations_f,
                    x_collocation,
                    grad_outputs=torch.ones_like(concentrations_f),
                    create_graph=True
                )[0]

                f_dC_drho = f_first_derivatives[:,:1]
                f_dC_dtau = f_first_derivatives[:,1:2]

                f_second_derivatives = torch.autograd.grad(
                    f_dC_drho,
                    x_collocation,
                    grad_outputs=torch.ones_like(f_dC_drho),
                    create_graph=True
                )[0]

                f_d2C_drho2 = f_second_derivatives[:,:1]

                f_rho = x_collocation[:,:1]
                residual_PDE = (f_dC_dtau - (f_d2C_drho2 + (2/f_rho) * f_dC_drho))
                batch_PDE_loss = torch.mean(residual_PDE.square())

                #BOUNDARY
                b_first_derivative = torch.autograd.grad(
                    concentrations_b,
                    x_boundary,
                    grad_outputs=torch.ones_like(concentrations_b),
                    create_graph=True
                )[0]

                b_dC_drho = b_first_derivative[:,:1]

                residual_boundary = -b_dC_drho - flux
                batch_BOUNDARY_loss = torch.mean(residual_boundary.square())

                #DATA
                error = concentrations_d - dy_train
                batch_DATA_loss = torch.mean(error.square())

                #LOSS
                batch_train_loss = lambda_d * batch_DATA_loss + lambda_f * batch_PDE_loss + lambda_b * batch_BOUNDARY_loss

                output_statistics = self.check_output_scale(concentrations_d)
                pde_statistics = self.check_pde_residual(residual_PDE)

                batch_train_loss.backward()

                gradient_statistics = self.check_gradients(model,batch_train_loss.item())

                opt1.step()

                if batch_index == 0:
                    print(
                        "Diagnostics | "
                        f"Output mean: {output_statistics['mean']:.3e} | "
                        f"Output range: "
                        f"[{output_statistics['minimum']:.3e}, "
                        f"{output_statistics['maximum']:.3e}] | "
                        f"PDE RMS: {pde_statistics['rms']:.3e} | "
                        f"Gradient norm: "
                        f"{gradient_statistics['total_norm']:.3e}"
                    )

                data_loss_sum += batch_DATA_loss.item() * dx_train.shape[0]
                pde_loss_sum += batch_PDE_loss.item() * fx_train.shape[0]
                boundary_loss_sum += batch_BOUNDARY_loss.item() * bx_train.shape[0]

                number_data_points += dx_train.shape[0]
                number_collocation_points += fx_train.shape[0]
                number_boundary_points += bx_train.shape[0]

            epoch_data_loss = (data_loss_sum / number_data_points)
            epoch_pde_loss = (pde_loss_sum / number_collocation_points)
            epoch_boundary_loss = (boundary_loss_sum / number_boundary_points)
            epoch_total_loss = (lambda_d * epoch_data_loss + lambda_f * epoch_pde_loss + lambda_b * epoch_boundary_loss)

            epoch_data_val_loss, epoch_pde_val_loss, epoch_boundary_val_loss, epoch_total_val_loss = self.validate(
                model,
                init_concentration,
                flux,
                lambda_d,
                lambda_f,
                lambda_b
                )
            
            print(
                f"Epoch {epoch + 1:5d}/{epochs} | "
                f"Total_train_loss: {epoch_total_loss:.4e} | "
                f"Total_val_loss: {epoch_total_val_loss:.4e} | "
                f"Data_train_loss: {epoch_data_loss:.4e} | "
                f"Data_val_loss: {epoch_data_val_loss:.4e} | "
                f"PDE_train_loss: {epoch_pde_loss:.4e} | "
                f"PDE_val_loss: {epoch_pde_val_loss:.4e} | "
                f"BC_train_loss: {epoch_boundary_loss:.4e} | "
                f"BC_val_loss: {epoch_boundary_val_loss:.4e} | "
                f"LR: {opt1.param_groups[0]['lr']:.2e} |"
                f"OPT: ADAM"
                )
            print("-" * 100)

            sheduler.step(epoch_total_val_loss)

            if epoch_data_val_loss < best_data_val_loss:
                best_data_val_loss = epoch_data_val_loss
                best_total_val_loss = epoch_total_val_loss
                patience = 0

                best_model_state = {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                    }
            else:
                if patience < early_stopping_patience:
                    patience += 1

            if patience >= early_stopping_patience:
                plateau_reached = True
                print("Plateau reached: switching from Adam to LBFGS.")
                break

        model.load_state_dict(best_model_state)
        final_data_validation_loss = best_data_val_loss
        final_total_validation_loss = best_total_val_loss
        final_optimizer = "ADAM"

        if plateau_reached:
            model = self.refine(
                model,
                init_concentration,
                flux,
                lambda_d,
                lambda_f,
                lambda_b,
                max_iter=lbfgs_max_iter
                )

            final_data_validation_loss, _, _, final_total_validation_loss = self.validate(
                model,
                init_concentration,
                flux,
                lambda_d,
                lambda_f,
                lambda_b
                )

            final_optimizer = "LBFGS"

            if final_data_validation_loss > best_data_val_loss:
                print(
                    "LBFGS worsened data validation loss: "
                    "restoring best Adam model."
                    )

                model.load_state_dict(best_model_state)
                final_data_validation_loss = best_data_val_loss
                final_total_validation_loss = best_total_val_loss
                final_optimizer = "ADAM"

        self.save_checkpoint(
            model=model,
            data_validation_loss=final_data_validation_loss,
            total_validation_loss=final_total_validation_loss,
            init_concentration=init_concentration,
            flux=flux,
            lambda_d=lambda_d,
            lambda_f=lambda_f,
            lambda_b=lambda_b,
            optimizer_name=final_optimizer
            )

        return model

    def validate(
            self,
            model,
            init_concentration,
            flux,
            lambda_d,
            lambda_f,
            lambda_b
            ):
        """Evaluate data, PDE, boundary, and weighted total validation losses."""

        model.eval()

        data_val_loss_sum = 0.0
        pde_val_loss_sum = 0.0
        boundary_val_loss_sum = 0.0

        number_data_points = 0
        number_collocation_points = 0
        number_boundary_points = 0

        for (dx_val,dy_val),(fx_val,),(bx_val,) in zip(self.d_val_loader,self.f_val_loader,self.b_val_loader):
            dx_val = dx_val.to(self.device)
            dy_val = dy_val.to(self.device)
            fx_val = fx_val.to(self.device)
            bx_val = bx_val.to(self.device)

            x_val_collocation = fx_val.detach().clone().requires_grad_(True)
            x_val_boundary = bx_val.detach().clone().requires_grad_(True)

            val_concentration_data = model(dx_val,init_concentration)
            val_concentration_f = model(x_val_collocation,init_concentration)
            val_concentration_b = model(x_val_boundary,init_concentration)

            #PDE
            val_f_fisrt_derivatives = torch.autograd.grad(
                outputs=val_concentration_f,
                inputs=x_val_collocation,
                grad_outputs=torch.ones_like(val_concentration_f),
                create_graph=True
            )[0]

            f_val_dC_drho,f_val_dC_dtau = val_f_fisrt_derivatives[:,:1],val_f_fisrt_derivatives[:,1:2]

            val_f_second_derivatives = torch.autograd.grad(
                outputs=f_val_dC_drho,
                inputs=x_val_collocation,
                grad_outputs=torch.ones_like(f_val_dC_drho),
                create_graph=False
            )[0]

            f_val_d2C_drho2 = val_f_second_derivatives[:,:1]

            f_val_rho = x_val_collocation[:,:1]
            f_val_resiudal = f_val_dC_dtau - (f_val_d2C_drho2 + (2/f_val_rho) * f_val_dC_drho)
            batch_pde_val_loss = torch.mean(f_val_resiudal.square())

            #BOUNDARY
            val_b_first_derivatives = torch.autograd.grad(
                outputs=val_concentration_b,
                inputs=x_val_boundary,
                grad_outputs=torch.ones_like(val_concentration_b),
                create_graph=False
            )[0]

            b_val_dC_drho = val_b_first_derivatives[:,:1]

            b_val_residual = -b_val_dC_drho - flux
            batch_boundary_val_loss = torch.mean(b_val_residual.square())

            #DATA
            batch_data_val_loss = torch.mean((val_concentration_data - dy_val).square())

            data_val_loss_sum += (batch_data_val_loss.item() * dx_val.shape[0])
            pde_val_loss_sum += (batch_pde_val_loss.item() * fx_val.shape[0])
            boundary_val_loss_sum += (batch_boundary_val_loss.item() * bx_val.shape[0])

            number_data_points += dx_val.shape[0]
            number_collocation_points += fx_val.shape[0]
            number_boundary_points += bx_val.shape[0]

        data_val_loss = (data_val_loss_sum / number_data_points)
        pde_val_loss = (pde_val_loss_sum / number_collocation_points)
        boundary_val_loss = (boundary_val_loss_sum / number_boundary_points)

        val_loss = (lambda_d * data_val_loss + lambda_f * pde_val_loss + lambda_b * boundary_val_loss)

        return data_val_loss, pde_val_loss, boundary_val_loss, val_loss

    def refine(
            self,
            model,
            init_concentration,
            flux,
            lambda_d,
            lambda_f,
            lambda_b,
            max_iter=500
            ):
        """Refine a trained PINN on full batches with L-BFGS and return it."""

        x_data, y_data = self.d_loader.dataset.tensors
        (x_collocation_base,) = self.f_loader.dataset.tensors
        (x_boundary_base,) = self.b_loader.dataset.tensors

        x_data = x_data.to(self.device)
        y_data = y_data.to(self.device)
        x_collocation_base = x_collocation_base.to(self.device)
        x_boundary_base = x_boundary_base.to(self.device) 

        opt2 = torch.optim.LBFGS(
            params=model.parameters(),
            lr=1.0,
            max_iter=max_iter,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            history_size=100,
            line_search_fn="strong_wolfe"
            )

        latest_losses = {}
        closure_calls = 0

        def closure():
            nonlocal closure_calls
            closure_calls += 1

            model.train()
            opt2.zero_grad()   

            x_collocation = x_collocation_base.detach().clone().requires_grad_(True)
            x_boundary = x_boundary_base.detach().clone().requires_grad_(True)

            concentrations_d = model(x_data,init_concentration)                                                            
            concentrations_f = model(x_collocation,init_concentration)                                                        
            concentrations_b = model(x_boundary,init_concentration)                                                           

            #PDE
            f_first_derivatives = torch.autograd.grad(
                concentrations_f,
                x_collocation,
                grad_outputs=torch.ones_like(concentrations_f),
                create_graph=True
            )[0]

            f_dC_drho = f_first_derivatives[:,:1]
            f_dC_dtau = f_first_derivatives[:,1:2]

            f_second_derivatives = torch.autograd.grad(
                f_dC_drho,
                x_collocation,
                grad_outputs=torch.ones_like(f_dC_drho),
                create_graph=True
            )[0]

            f_d2C_drho2 = f_second_derivatives[:,:1]

            f_rho = x_collocation[:,:1]
            residual_PDE = (f_dC_dtau - (f_d2C_drho2 + (2/f_rho) * f_dC_drho))
            PDE_loss = torch.mean(residual_PDE.square())

            #BOUNDARY
            b_first_derivative = torch.autograd.grad(
                concentrations_b,
                x_boundary,
                grad_outputs=torch.ones_like(concentrations_b),
                create_graph=True
            )[0]

            b_dC_drho = b_first_derivative[:,:1]

            residual_boundary = -b_dC_drho - flux
            BOUNDARY_loss = torch.mean(residual_boundary.square())

            #DATA
            error = concentrations_d - y_data
            DATA_loss = torch.mean(error.square())

            #LOSS
            train_loss = lambda_d * DATA_loss + lambda_f * PDE_loss + lambda_b * BOUNDARY_loss

            train_loss.backward()

            latest_losses["total"] = train_loss.detach().item()
            latest_losses["data"] = DATA_loss.detach().item()
            latest_losses["pde"] = PDE_loss.detach().item()
            latest_losses["boundary"] = (BOUNDARY_loss.detach().item())

            return train_loss



        opt2.step(closure)
        optimizer_closure_calls = closure_calls

        #evaluate training loss on final parameters
        closure()
        opt2.zero_grad(set_to_none=True)

        data_val_loss, pde_val_loss, b_val_loss, val_loss = self.validate(
            model,
            init_concentration,
            flux,
            lambda_d,
            lambda_f,
            lambda_b
            )
        
        print(
            "LBFGS refinement completed | "
            f"Max iterations: {max_iter} | "
            f"Closure evaluations: {optimizer_closure_calls} | "
            f"Total train: {latest_losses['total']:.4e} | "
            f"Total validation: {val_loss:.4e} | "
            f"Data train: {latest_losses['data']:.4e} | "
            f"Data validation: {data_val_loss:.4e} | "
            f"PDE train: {latest_losses['pde']:.4e} | "
            f"PDE validation: {pde_val_loss:.4e} | "
            f"BC train: {latest_losses['boundary']:.4e} | "
            f"BC validation: {b_val_loss:.4e} | "
            "OPT: LBFGS"
        )
        print("-" * 100)

        return model

    def save_checkpoint(
            self,
            model,
            data_validation_loss,
            total_validation_loss,
            init_concentration,
            flux,
            lambda_d,
            lambda_f,
            lambda_b,
            optimizer_name,
            output_folder="./Results/Models",
            filename="pinn_checkpoint.pth"
            ):

        output_folder = Path(output_folder)
        output_folder.mkdir(
            parents=True,
            exist_ok=True
            )

        checkpoint_path = output_folder / filename

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "data_validation_loss": float(data_validation_loss),
            "total_validation_loss": float(total_validation_loss),
            "initial_concentration": float(init_concentration),
            "flux": float(flux),
            "lambda_data": float(lambda_d),
            "lambda_pde": float(lambda_f),
            "lambda_boundary": float(lambda_b),
            "optimizer": optimizer_name
            }

        torch.save(
            checkpoint,
            checkpoint_path
            )

        print(
            "Model checkpoint saved to: "
            f"{checkpoint_path.resolve()}"
            )

        return checkpoint_path

    def check_output_scale(self,concentrations: torch.Tensor) -> dict[str, float]:
        """Validate finite concentration outputs and return scale statistics."""
        values = concentrations.detach()

        if not torch.isfinite(values).all():
            raise FloatingPointError(
                "NaN or Inf detected in model outputs."
            )

        statistics = {
            "minimum": values.min().item(),
            "maximum": values.max().item(),
            "mean": values.mean().item(),
            "std": values.std(unbiased=False).item(),
            "maximum_absolute": values.abs().max().item(),
        }

        if (
            statistics["minimum"] < -0.05
            or statistics["maximum"] > 1.05
        ):
            warnings.warn(
                "Concentration predictions are outside "
                "the expected physical range [0, 1]."
            )

        if statistics["maximum_absolute"] > 10.0:
            warnings.warn(
                "Model outputs have an unexpectedly large scale."
            )

        return statistics

    def check_pde_residual(self,residual: torch.Tensor,baseline_rms: float | None = None) -> dict[str, float]:
        """Validate a PDE residual tensor and return magnitude statistics."""
        values = residual.detach()

        if not torch.isfinite(values).all():
            raise FloatingPointError(
                "NaN or Inf detected in PDE residual."
            )

        rms = values.square().mean().sqrt().item()

        statistics = {
            "mean_absolute": values.abs().mean().item(),
            "rms": rms,
            "maximum_absolute": values.abs().max().item(),
        }

        if rms > 1e2:
            warnings.warn(
                f"Large PDE residual RMS: {rms:.3e}."
            )

        if (
            baseline_rms is not None
            and rms > 100.0 * max(baseline_rms, 1e-12)
        ):
            warnings.warn(
                "PDE residual increased by more than "
                "two orders of magnitude from its baseline."
            )

        if statistics["maximum_absolute"] > 1e6:
            raise FloatingPointError(
                "PDE residual exploded."
            )

        return statistics

    def check_gradients(self,model: torch.nn.Module,loss_value: float) -> dict[str, object]:
        """Detect invalid, vanishing, or exploding gradients and summarize them."""
        squared_total_norm = 0.0
        maximum_absolute_gradient = 0.0
        layer_norms = {}

        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue

            gradient = parameter.grad.detach()

            if not torch.isfinite(gradient).all():
                raise FloatingPointError(
                    f"NaN or Inf detected in gradient: {name}."
                )

            gradient_norm = gradient.norm(2).item()
            layer_norms[name] = gradient_norm

            squared_total_norm += gradient_norm**2

            maximum_absolute_gradient = max(
                maximum_absolute_gradient,
                gradient.abs().max().item(),
            )

        total_norm = squared_total_norm**0.5

        if total_norm > 1e3:
            warnings.warn(
                f"Possible exploding gradients: "
                f"norm={total_norm:.3e}."
            )

        if total_norm < 1e-10 and loss_value > 1e-8:
            warnings.warn(
                f"Possible vanishing gradients: "
                f"norm={total_norm:.3e}."
            )

        input_gradient = layer_norms.get("input.weight")
        output_gradient = layer_norms.get("out.weight")

        if (
            input_gradient is not None
            and output_gradient is not None
            and output_gradient > 0.0
            and input_gradient / output_gradient < 1e-6
        ):
            warnings.warn(
                "Input-layer gradients are much smaller than "
                "output-layer gradients: possible vanishing gradient."
            )

        return {
            "total_norm": total_norm,
            "maximum_absolute": maximum_absolute_gradient,
            "layer_norms": layer_norms,
        }

    def save_checkpoint(
        self,
        model,
        data_validation_loss,
        total_validation_loss,
        init_concentration,
        flux,
        lambda_d,
        lambda_f,
        lambda_b,
        optimizer_name,
        output_folder="./Results/Models",
        filename="pinn_checkpoint.pth",
    ):
        output_folder = Path(output_folder)
        output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename_path = Path(filename)

        checkpoint_path = output_folder / (f"{filename_path.stem}_{timestamp}"
                                            f"{filename_path.suffix}")

        collision_index = 1
        while checkpoint_path.exists():
            checkpoint_path = output_folder / (
                                                f"{filename_path.stem}_{timestamp}_"
                                                f"{collision_index}{filename_path.suffix}"
                                                )
            collision_index += 1

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "data_validation_loss": float(data_validation_loss),
            "total_validation_loss": float(total_validation_loss),
            "initial_concentration": float(init_concentration),
            "flux": float(flux),
            "lambda_data": float(lambda_d),
            "lambda_pde": float(lambda_f),
            "lambda_boundary": float(lambda_b),
            "optimizer": optimizer_name,
        }

        torch.save(
            checkpoint,
            checkpoint_path,
        )

        print(
            f"Model checkpoint saved to: "
            f"{checkpoint_path.resolve()}"
        )

        return checkpoint_path
