## [0.4.3] - 2024-01-05
### Changed
- Moded `tensorBoardLogger` in `rockrl.tensorflow.ppo` few lines bellow

## [0.4.3] - 2024-01-05
### Changed
- Updated `rockrl.tensorflow.ppo` to use threading while logging, to save time

## [0.4.2] - 2024-01-02
### Changed
- Changed `rockrl.utils.VectorizedEnv` to return info dictionaries in list instead of tuples


## [0.4.1] - 2023-12-08
### Changed
- Removed `tensorflow_probability` dependency


## [0.4.0] - 2023-10-10
### Added
- Initial RockRL package release
- Tensorflow PPO implementation for `discrete` and `continuous` action spaces
- Added `Memory` (one environment) and `MemoryHandler` (multiple environments) object to `rockrl.utils` package
- Examples for `LunarLander-v2`, `BipedalWalker-v3` and `BipedalWalkerHardcore-v3` environments