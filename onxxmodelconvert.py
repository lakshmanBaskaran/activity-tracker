import joblib
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

# Load model and define input shape
model = joblib.load("models/simple_lgbm_model.joblib")

# Adjust input feature count if needed (C * 7 from your feature extraction)
n_features = 42  # 6 channels × 7 features each
initial_type = [('input', FloatTensorType([None, n_features]))]

# Convert to ONNX
onnx_model = onnxmltools.convert_lightgbm(model, initial_types=initial_type)
onnxmltools.utils.save_model(onnx_model, "models/simple_lgbm_model.onnx")
print("Converted to ONNX: models/simple_lgbm_model.onnx")
