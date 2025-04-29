# Atari Game Environments

![Atari Game](../images/ms_pacman.gif)

This project uses environments from the Atari collection in Gymnasium. Each environment has its own specific characteristics.

## Action Space
Action spaces vary by game, typically ranging from Discrete(4) to Discrete(18).

## Observation Space
Typically Box(0, 255, (210, 160, 3), uint8) for RGB observations.

## Import
```python
gymnasium.make(config.ENV_NAME)  # Uses the environment name from config
```

For different variants with different observation and action spaces, see the Gymnasium documentation.

## Description

Each Atari game has its own goal and mechanics. The agent must learn to maximize its score by understanding the rules and dynamics of the specific game.

For more detailed documentation on specific games, see the Gymnasium documentation or AtariAge pages.

## Actions

Atari games typically have action spaces from 4 to 18 possible actions, with the following common ones:

| Common Actions | Meaning   |
|---------------|-----------|
| NOOP          | No operation |
| FIRE          | Fire button |
| UP/RIGHT/LEFT/DOWN | Directional movements |
| Combined directions | Diagonal movements |

The action space size is defined in the configuration file and may vary based on the selected game.

## Observations

Atari environments have three possible observation types:

- `obs_type="rgb"` -> `observation_space=Box(0, 255, (210, 160, 3), np.uint8)`
- `obs_type="ram"` -> `observation_space=Box(0, 255, (128,), np.uint8)`
- `obs_type="grayscale"` -> `Box(0, 255, (210, 160), np.uint8)`, a grayscale version of the "rgb" type

Our implementation uses preprocessing to standardize these observations.

## Variants

Atari games have multiple variants with the following differences:
- Observation type (rgb/ram/grayscale)
- Frame-skips (how many frames to skip between actions)
- Action repeat probability

Example naming patterns:
- `GameName-v0`: Original version with frame skipping (2-5) and 0.25 repeat probability
- `GameName-ram-v0`: RAM observations instead of visual
- `GameNameDeterministic-v0`: Fixed 4-frame skip
- `GameNameNoFrameskip-v0`: No frame skipping (1 frame)
- `GameName-v4`: Same as v0 but with 0.0 repeat probability
- `ALE/GameName-v5`: Latest version (ALE namespace) with 4-frame skip and 0.25 repeat probability

## Difficulty and modes

Many Atari games support various difficulty levels and game modes that can be specified via the `difficulty` and `mode` parameters when creating the environment. These are defined in the configuration file.

## Version History

Atari environments in Gymnasium have evolved through several versions:

- v5: Action stickiness was reintroduced and stochastic frame-skipping was removed. Environments moved to the "ALE" namespace.
- v4: Stickiness of actions was removed
- v0: Initial versions release

For detailed differences between versions, refer to the Gymnasium documentation on Atari environments.