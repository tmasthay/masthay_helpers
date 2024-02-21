import hydra
import pytest

from masthay_helpers.global_helpers import hydra_kw

with hydra.initialize(config_path='config', version_base=None):
    cfg = hydra.compose(config_name='config')
param = pytest.mark.parametrize

for i, e in enumerate(cfg.hydra_kw.cases):
    cfg.hydra_kw.cases[i][-1] += cfg.hydra_kw.dummy_var


@hydra_kw(use_cfg=False, protect_kw=True)
def add(x, y, dummy_var=0):
    return x + y + dummy_var


@param(cfg.hydra_kw.name, cfg.hydra_kw.cases)
def test_hydra_kw(x, y, expected_output):
    assert add(x, y, dummy_var=cfg.hydra_kw.dummy_var) == expected_output


if __name__ == "__main__":
    print(f'Not passing in dummy_var\n{80*"*"}')
    for i, e in enumerate(cfg.hydra_kw.cases):
        print(f'{e[0]} + {e[1]} + {cfg.hydra_kw.dummy_var} = {add(e[0], e[1])}')
    print(80 * '*')
    dummy_var = 100
    print(f'\nPassing in dummy_var={dummy_var}\n{80*"*"}')
    for e in cfg.hydra_kw.cases:
        print(
            f'{e[0]} + {e[1]} +'
            f' {dummy_var} = {add(e[0], e[1], dummy_var=dummy_var)}'
        )
    print(80 * '*')
