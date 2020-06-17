from data import MetaLearningSystemDataLoader
from experiment_builder import ExperimentBuilder
from meta_learning_system import SceneAdaptiveInterpolation
from config import get_args

args, _ = get_args()
print(args)
model = SceneAdaptiveInterpolation(args)
data = MetaLearningSystemDataLoader
savfi_system = ExperimentBuilder(model=model, data=data, args=args)
savfi_system.run_experiment()