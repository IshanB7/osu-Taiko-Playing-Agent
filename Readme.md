# osu-Taiko Playing Agent

## Overview

The `auto_best.py` script can be run concurrently with the `screen_capture.py` script to help generate training data.

- The `screen_capture.py` script cannot detect key release events. As a result, the generated data will not include the `'-'` label. (see how this works )

- The `agent_auto_capture.py` script is intended to incorporate both `screen_capture.py` and `agent_best.py`. However, it currently results in significantly more missing key presses.