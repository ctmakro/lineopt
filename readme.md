# LineOpt

I don't have time explain what these code does.

## Dependencies

- <https://github.com/ctmakro/cv2tools>
- opencv-python
- numpy and stuff

## Give it a taste

```bash
$ ipython -i lineopt.py jeff.jpg
>>> schedule3()
```

## Order of testing

1. connect via UGS and home.

2. feed `burnout.gcode` to make sure the bot works well with current setting in extreme conditions.

3. `python toolchange.py` to test the tool change

4. (for brushes) `python paintstation.py` to test the paint and water stations
