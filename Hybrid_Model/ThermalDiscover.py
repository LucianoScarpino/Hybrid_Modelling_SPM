import torch
import pysindy as ps
import pandas as pd
import numpy as np

from copy import deepcopy
from pathlib import Path
from time import perf_counter
from scipy.integrate import solve_ivp
from torch.utils.data import TensorDataset,DataLoader
from sklearn.metrics import root_mean_squared_error,mean_absolute_error
from sklearn.preprocessing import StandardScaler

from NeuralNetwork import ThermalFFN


def compute_thermal_metrics(
        reference_temperature,
        predicted_temperature,
        current,
        time,
        T_amb=298.15
        ):
    reference_temperature = np.asarray(reference_temperature, dtype=float)
    predicted_temperature = np.asarray(predicted_temperature, dtype=float)
    current = np.asarray(current, dtype=float)
    time = np.asarray(time, dtype=float)

    error = predicted_temperature - reference_temperature
    reference_theta = reference_temperature - T_amb
    predicted_theta = predicted_temperature - T_amb
    theta_norm = np.linalg.norm(reference_theta)

    peak_current_mask = np.isclose(current, np.max(current))
    peak_current_rmse = root_mean_squared_error(
        reference_temperature[peak_current_mask],
        predicted_temperature[peak_current_mask]
    )

    active_indices = np.flatnonzero(~np.isclose(current, 0.0))
    if len(active_indices) > 0 and active_indices[-1] < len(current) - 1:
        cooling_mask = np.arange(len(current)) > active_indices[-1]
        cooling_rmse = root_mean_squared_error(
            reference_temperature[cooling_mask],
            predicted_temperature[cooling_mask]
        )
    else:
        cooling_rmse = float("nan")

    reference_peak_index = int(np.argmax(reference_temperature))
    predicted_peak_index = int(np.argmax(predicted_temperature))

    return {
        "temperature_rmse": root_mean_squared_error(
            reference_temperature,
            predicted_temperature
        ),
        "temperature_mae": mean_absolute_error(
            reference_temperature,
            predicted_temperature
        ),
        "temperature_max_error": float(np.max(np.abs(error))),
        "relative_l2_theta": (
            float(np.linalg.norm(predicted_theta - reference_theta) / theta_norm)
            if theta_norm > np.finfo(float).eps else float("nan")
        ),
        "peak_temperature_error": float(
            abs(np.max(predicted_temperature) - np.max(reference_temperature))
        ),
        "peak_time_error": float(
            abs(time[predicted_peak_index] - time[reference_peak_index])
        ),
        "peak_current_rmse": peak_current_rmse,
        "cooling_rmse": cooling_rmse,
        "final_temperature_error": float(abs(error[-1])),
    }

class SINDyc(object):
    def __init__(self,reference_dataset):
        self.reference_dataset = reference_dataset

    @staticmethod
    def _signals(dataset,T_amb):
        signals = (
            dataset
            .drop_duplicates("time")
            .sort_values("time")
            .reset_index(drop=True)
        )

        time = signals["time"].to_numpy(dtype=float)
        temperature = signals["temperature"].to_numpy(dtype=float)
        current = signals["current"].to_numpy(dtype=float)
        temperature_state = (temperature - T_amb).reshape(-1,1)
        control_input = current.reshape(-1,1)

        return signals,time,temperature,current,temperature_state,control_input

    def discover_equation(
            self,
            threshold_range,
            validation_dataset=None,
            T_amb=298.15
            ):
        training_datasets = (
            list(self.reference_dataset)
            if isinstance(self.reference_dataset, (list, tuple))
            else [self.reference_dataset]
        )

        training_signals = [
            self._signals(dataset, T_amb)
            for dataset in training_datasets
        ]

        temperature_states = [signals[4] for signals in training_signals]
        time_grids = [signals[1] for signals in training_signals]
        control_inputs = [signals[5] for signals in training_signals]

        selection_dataset = (
            validation_dataset
            if validation_dataset is not None
            else training_datasets[0]
        )
        (
            _,selection_time,selection_temperature,selection_current,
            selection_state,_
        ) = self._signals(selection_dataset, T_amb)

        library = ps.PolynomialLibrary(degree=2,include_bias=True)
        differentiation = ps.FiniteDifference(order=2,drop_endpoints=True)

        candidates = []

        for threshold in threshold_range:
            optimizer = ps.STLSQ(
                threshold=threshold,
                alpha=1e-8,
                normalize_columns=True
            )

            model = ps.SINDy(
                feature_library=library,
                optimizer=optimizer,
                differentiation_method=differentiation
            )

            if len(training_datasets) == 1:
                model.fit(
                    temperature_states[0],
                    t=time_grids[0],
                    u=control_inputs[0]
                )
            else:
                model.fit(
                    temperature_states,
                    t=time_grids,
                    u=control_inputs
                )
            print(f"SINDy Running...")
            print(f"threshold: {threshold}")
            model.print(lhs=["dtheta/dt"])

            pred = self._predict_temperatures(
                model,
                selection_state,
                T_amb,
                selection_time,
                selection_current
            )

            metrics = compute_thermal_metrics(
                selection_temperature,
                pred,
                selection_current,
                selection_time,
                T_amb
            )

            print(f"RMSE: {metrics['temperature_rmse']:.4e} |"
                  f"MAE: {metrics['temperature_mae']:.4e} |"
                  f"Max Error: {metrics['temperature_max_error']:.4e} |"
                  f"Peak Temperature Error: {metrics['peak_temperature_error']:.4e} |"
                  f"Peak-current RMSE: {metrics['peak_current_rmse']:.4e} |"
                  f"Cooling RMSE: {metrics['cooling_rmse']:.4e} |"
                  )

            print("-"*100)

            complexity = model.complexity

            candidates.append(
                {
                    "model":model,
                    "predictions":pred,
                    "threshold":threshold,
                    "RMSE":metrics["temperature_rmse"],
                    "MAE":metrics["temperature_mae"],
                    "complexity":complexity
                }
            )

            print(f"Complexity: {complexity}")

        minimum_RMSE = min(
            candidate["RMSE"]
            for candidate in candidates
        )

        RMSE_tolerance = 0.15

        maximum_acceptable_RMSE = (
            minimum_RMSE
            * (1.0 + RMSE_tolerance)
        )

        acceptable_candidates = [
            candidate
            for candidate in candidates
            if candidate["RMSE"] <= maximum_acceptable_RMSE
        ]

        best_candidate = min(
            acceptable_candidates,
            key=lambda candidate:(
                candidate["complexity"],
                candidate["RMSE"]
            )
        )

        best_model = best_candidate["model"]
        best_pred_temperatures = best_candidate["predictions"]
        best_threshold = best_candidate["threshold"]
        best_RMSE = best_candidate["RMSE"]
        best_complexity = best_candidate["complexity"]

        print(f"SINDy completed.")
        print(f"Minimum RMSE: {minimum_RMSE:.6e}")
        print(f"Maximum acceptable RMSE: {maximum_acceptable_RMSE:.6e}")
        print(f"Threshold selected: {best_threshold}")
        print(f"Selected RMSE: {best_RMSE:.6e}")
        print(f"Selected complexity: {best_complexity}")

        print("Discovered equation:")
        best_model.print(
            lhs=["dtheta/dt"],
            precision=8
        )

        return (
            best_model,
            best_pred_temperatures,
            best_threshold
        )

    def evaluate_dataset(self,model,dataset,T_amb=298.15):
        (
            _,time,temperature,current,temperature_state,control_input
        ) = self._signals(dataset, T_amb)

        rollout_start = perf_counter()
        predicted_temperature = self._predict_temperatures(
            model,
            temperature_state,
            T_amb,
            time,
            current
        )
        rollout_seconds = perf_counter() - rollout_start

        predicted_derivative = model.predict(
            temperature_state,
            u=control_input
        )[:,0]
        reference_derivative = np.gradient(
            temperature_state[:,0],
            time,
            edge_order=2
        )

        metrics = compute_thermal_metrics(
            temperature,
            predicted_temperature,
            current,
            time,
            T_amb
        )
        metrics.update({
            "derivative_rmse": root_mean_squared_error(
                reference_derivative,
                predicted_derivative
            ),
            "derivative_mae": mean_absolute_error(
                reference_derivative,
                predicted_derivative
            ),
            "rollout_seconds": rollout_seconds,
        })

        radius = np.sort(dataset["rho"].unique())

        return {
            "time": time,
            "radius": radius,
            "current": current,
            "reference_temperature": temperature,
            "predicted_temperature": predicted_temperature,
            "reference_temperature_map": np.tile(
                temperature[None,:],
                (len(radius),1)
            ),
            "predicted_temperature_map": np.tile(
                predicted_temperature[None,:],
                (len(radius),1)
            ),
            "reference_derivative": reference_derivative,
            "predicted_derivative": predicted_derivative,
            "metrics": metrics,
        }

    def _predict_temperatures(
            self,
            model,
            T_state,
            T_amb,
            time,
            current
            ):
        
        def current_function(current_time):
            interpolated_current = np.interp(
                current_time,
                time,
                current
            )

            return np.array([interpolated_current])

        predicted_state = model.simulate(
            T_state[0],
            time,
            u=current_function,
            integrator="solve_ivp",
            integrator_kws={
                "method":"LSODA",
                "rtol":1e-8,
                "atol":1e-10
            }
        )

        predicted_temperatures = predicted_state[:,0] + T_amb

        return predicted_temperatures

    def validate(self,preds,temperatures,current,time=None,T_amb=298.15):
        if time is None:
            time = np.arange(len(temperatures), dtype=float)

        metrics = compute_thermal_metrics(
            temperatures,
            preds,
            current,
            time,
            T_amb
        )

        return (
            metrics["temperature_rmse"],
            metrics["temperature_mae"],
            metrics["temperature_max_error"],
            metrics["peak_temperature_error"],
            metrics["peak_current_rmse"],
            metrics["cooling_rmse"]
        )

class FFNDiscovering(object):
    def __init__(self,source_path,datasets_names,T_amb=298.15):
        self.source_path = source_path
        self.datasets_names = datasets_names
        self.T_amb = T_amb

        self.input_scaler = None
        self.target_scaler = None

    def align_dataset(self,dataset):
        thermal_data = (
            dataset[["time","current","temperature"]]
            .drop_duplicates("time")
            .sort_values("time")
            .reset_index(drop=True)
            )

        thermal_data["theta"] = thermal_data["temperature"] - self.T_amb
        thermal_data["dtheta_dt"] = np.gradient(
            thermal_data["theta"].to_numpy(),
            thermal_data["time"].to_numpy(),
            edge_order=2
        )

        return thermal_data

    def get_training_data(self,datasets):
        return pd.concat(
            datasets,
            ignore_index=True
        )

    def generate_thermal_loaders(self):
        data_path = Path(self.source_path)

        datasets = []                               #8 datasets, 0-4 for training, 5 for validation, 6-7 for test

        for name in self.datasets_names:
            data_location = data_path / name
            data = pd.read_csv(data_location)

            aligned = self.align_dataset(data)
            datasets.append(aligned)



        input_scaler = StandardScaler()
        target_scaler = StandardScaler()

        training_data = self.get_training_data(datasets[0:5])
        val_data = datasets[5]
        interpolation_test_data = datasets[6]
        extrapolation_test_data = datasets[7]

        input_scaler.fit(training_data[["theta","current"]])
        target_scaler.fit(training_data[["dtheta_dt"]])

        def dataframe_to_tensor(data):
            X = torch.tensor(
                input_scaler.transform(data[["theta","current"]]),
                dtype=torch.float32,
            )

            y = torch.tensor(
                target_scaler.transform(data[["dtheta_dt"]]),
                dtype=torch.float32
            )

            return X,y

        x_train,y_train = dataframe_to_tensor(training_data)
        x_val,y_val = dataframe_to_tensor(val_data)
        x_int_test,y_int_test = dataframe_to_tensor(interpolation_test_data)
        x_ext_test,y_ext_test = dataframe_to_tensor(extrapolation_test_data)

        train_loader = DataLoader(
            TensorDataset(x_train,y_train),
            batch_size=512,
            shuffle=True
            )

        val_loader = DataLoader(
            TensorDataset(x_val,y_val),
            batch_size=512,
            shuffle=False
        )

        interpolation_test_loader = DataLoader(
            TensorDataset(x_int_test,y_int_test),
            batch_size=512,
            shuffle=False
        )

        extrapolation_test_loader = DataLoader(
            TensorDataset(x_ext_test,y_ext_test),
            batch_size=512,
            shuffle=False
        )

        self.input_scaler = input_scaler
        self.target_scaler = target_scaler

        return (
            train_loader,
            val_loader,
            interpolation_test_loader,
            extrapolation_test_loader,
        )

    def predict_equations(
            self,
            device,
            lr,
            training_epochs,
            early_stopping_epochs,
            train_loader,
            val_loader
            ):

        model = ThermalFFN(device).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)

        best_val_loss = float("inf")
        best_params = None
        patience = 0

        print("Training Thermal FFN...")

        for epoch in range(training_epochs):
            model.train()

            batch_loss = 0.0
            num_points = 0

            for x,y in train_loader:
                optimizer.zero_grad()

                x = x.to(device)
                y = y.to(device)

                derivative = model(x)
                res = torch.mean((derivative - y).square())

                res.backward()
                optimizer.step()

                batch_loss += res.item() * x.shape[0]
                num_points += x.shape[0]

            train_loss = batch_loss/num_points

            val_batch_loss = 0.0
            num_val_points = 0

            for val_x,val_y in val_loader:
                model.eval()

                val_x = val_x.to(device)
                val_y = val_y.to(device)

                with torch.no_grad():
                    val_derivative = model(val_x)
                    val_res = torch.mean((val_derivative - val_y).square())

                    val_batch_loss += val_res.item() * val_x.shape[0]
                    num_val_points += val_x.shape[0]

            val_loss = val_batch_loss/num_val_points

            if (
                epoch == 0
                or (epoch + 1) % 100 == 0
                or epoch == training_epochs - 1
            ):
                print(
                    f"[{epoch + 1}/{training_epochs}] "
                    f"train loss: {train_loss:.6e} | "
                    f"val loss: {val_loss:.6e}"
                )
                print("-"*100)

            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_params = deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= early_stopping_epochs:
                    print(f"Early stopping at epoch: {epoch}")
                    break

        model.load_state_dict(best_params)
        model.eval()

        return model

    def test_thermalFNN(
            self,
            model,
            test_loader,
            device,
            test_type,
            T_amb=298.15,
            ):

        if test_type not in {"interpolation", "extrapolation"}:
            raise ValueError(
                "test_type must be 'interpolation' or 'extrapolation'."
            )
        
        dataset_index = {
            "interpolation": 6,
            "extrapolation": 7
        }[test_type]

        test_path = Path(self.source_path)/self.datasets_names[dataset_index]
        raw_test_data = pd.read_csv(test_path)

        radius = np.sort(
            raw_test_data["rho"].unique()
        )

        test_data = self.align_dataset(raw_test_data)

        time = test_data["time"].to_numpy(dtype=float)
        current = test_data["current"].to_numpy(dtype=float)
        reference_temperature = test_data["temperature"].to_numpy(dtype=float)

        test_loss = 0.0
        num_test_points = 0

        dtheta_dt = []
        targets = []

        model.eval()

        for test_x,test_y in test_loader:
            test_x = test_x.to(device)
            test_y = test_y.to(device)

            with torch.no_grad():
                pred = model(test_x)
                res = torch.mean((pred - test_y).square())

                test_loss += res.item() * test_x.shape[0]
                num_test_points += test_x.shape[0]

                dtheta_dt.append(self.target_scaler.inverse_transform(pred.detach().cpu().numpy()))
                targets.append(self.target_scaler.inverse_transform(test_y.detach().cpu().numpy()))

        loss = test_loss/num_test_points

        predictions = np.concatenate(dtheta_dt,axis=0)
        targets = np.concatenate(targets,axis=0)

        physical_loss = np.mean((predictions - targets)**2)

        rollout_start = perf_counter()
        predicted_temperatures = self.temperature_rollout(
            model,
            time,
            current,
            reference_temperature[0],
            device=device,
            T_amb=T_amb
        )
        rollout_seconds = perf_counter() - rollout_start

        metrics = compute_thermal_metrics(
            reference_temperature,
            predicted_temperatures,
            current,
            time,
            T_amb
        )
        metrics.update({
            "normalized_derivative_mse": loss,
            "physical_derivative_mse": physical_loss,
            "derivative_rmse": root_mean_squared_error(targets,predictions),
            "derivative_mae": mean_absolute_error(targets,predictions),
            "rollout_seconds": rollout_seconds,
        })

        reference_temperature_map = np.tile(
            reference_temperature[None, :],
            (len(radius), 1)
        )

        predicted_temperature_map = np.tile(
            predicted_temperatures[None, :],
            (len(radius), 1)
        )

        print(
            f"Test type: {test_type} |"
            f"Test loss: {loss:.6e} |"
            f"Physical loss: {physical_loss:.6e} |"
            f"temperature RMSE: {metrics['temperature_rmse']:.6e} |"
            f"temperature MAE: {metrics['temperature_mae']:.6e} |"
            f"temperature max error: {metrics['temperature_max_error']:.6e} |"
            f"rollout time: {rollout_seconds:.6e} s |"
        )

        return {
            "test_type": test_type,

            "time": time,
            "radius": radius,
            "current": current,

            "reference_temperature": reference_temperature,
            "predicted_temperature": predicted_temperatures,

            "reference_temperature_map":
                reference_temperature_map,

            "predicted_temperature_map":
                predicted_temperature_map,

            "reference_derivative": targets,
            "predicted_derivative": predictions,

            "metrics": metrics
        }

    def temperature_rollout(
            self,
            model,
            time,
            current,
            init_temperature,
            device,
            T_amb
            ):

        predicted_theta = np.zeros_like(time, dtype=float)
        predicted_theta[0] = init_temperature - T_amb

        model.eval().to(device)

        def thermal_rhs(current_time,state):
            physical_input = pd.DataFrame(
                [[state[0],np.interp(current_time,time,current)]],
                columns=["theta","current"]
            )
            model_input = torch.tensor(
                self.input_scaler.transform(physical_input),
                dtype=torch.float32,
                device=device
            )

            with torch.no_grad():
                normalized_derivative = (
                    model(model_input)
                    .detach()
                    .cpu()
                    .numpy()
                )

            physical_derivative = self.target_scaler.inverse_transform(
                normalized_derivative
            )[0,0]

            return np.array([physical_derivative],dtype=float)

        solution = solve_ivp(
            thermal_rhs,
            (time[0],time[-1]),
            [predicted_theta[0]],
            t_eval=time,
            method="LSODA",
            rtol=1e-8,
            atol=1e-10
        )

        if not solution.success:
            raise RuntimeError(
                f"FFN temperature rollout failed: {solution.message}"
            )

        predicted_theta = solution.y[0]

        return predicted_theta + T_amb
