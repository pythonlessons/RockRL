## [0.4.6] - 2024-01-30
### Added
- Added `ema_ratio` parameter to `rockrl.tensorflow.ppo` to control the exponential moving average ratio of the old and new networks

## [0.4.5] - 2024-01-17
### Changed
- Changed `rockrl.tensorflow.ppo` `act` function to use training mode while training and evaluation mode while evaluating

### Added
- Added `close()` method to `rockrl.tensorflow.ppo` to close `tensorBoardLogger` after training

## [0.4.4] - 2024-01-05
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