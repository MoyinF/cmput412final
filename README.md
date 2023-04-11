# Exercise 5: ML for Robotics

This repository contains implementation solutions for exercise 5. For information about the project, please read the report at:

<!-- TODO: add URLs -->

[Nadeen Mohamed's site]() or [Moyinoluwa Famobiw's site]() or [Austin Tralnberg's Site]()

## Structure

There are two packages in this file: duckiebot_detection and driver. We will discuss the purpose of the python source files for each package (which are located inside the packages `src` folder).

### Driver

<!-- TODO: add info about driver -->

### Duckiebot Detection

<!-- TODO: add info about duckiebot detection -->

## Execution:

To set the stall parameter, change the number in `/data/stall`, for example with the following steps:

```
ssh duckie@csc229xx.local # where csc229xx is the duckiebot's hostname
vim /data/stall # creates or opens the stall file, where you write the number of the stall and save
```

To run the program, ensure that the variable `$BOT` stores your robot's host name (ie. `csc229xx`), and run the following commands:

```
dts devel build -f -H $BOT.local
dts devel run -H $BOT.local
```

To shutdown the program, enter `CTRL + C` in your terminal.

## Credit:

This code is built from the Duckiebot detections tarter code by Zepeng Xiao (https://github.com/XZPshaw/CMPUT412503_exercise4).

Build on top of by Nadeen Mohamed, Moyinoluwa Famobiwo, and Austin Tralnberg.

Autonomous lane following code was also borrowed from Justin Francis.

Code was also borrowed (and cited in-code) from the following sources:

<!-- TODO: add code sources -->
