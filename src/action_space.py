"""
Action space definitions mapped directly to poke-env's native integer space.
"""

# Native poke-env Singles Action Space:
# 0-5: switch
# 6-9: move
# 10-13: move + mega evolve
# 14-17: move + z-move
# 18-21: move + dynamax
# (22-25: move + terastallize, if using Gen 9)
NATIVE_ACTION_SPACE_N = 22

def is_native_switch_action(action: int) -> bool:
    """Check if the native poke-env action is a switch (actions 0-5)."""
    return 0 <= int(action) <= 5