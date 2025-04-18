import math
import sys
import time
from enum import IntEnum

import chess
import numpy
import pyfirmata
from matplotlib import pyplot
from pyfirmata import util
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink


def start_iterator(board):
    util.Iterator(board).start()


# min and max angles to which the axles can move
AXLE_BOUNDS = [
    (-90, 90),  # shoulder
    (-90, 90),  # elbow
]

# angle by which the orientation is changed at each step of the axle motors
AXLE_STEP_ANGLES = [
    1.8 / 4,
    1.8 / 4,
]

# speed of steppers depends on these values
AXLE_DELAYS = [  # in microseconds, precision seems to be around 10 microseconds upon normal CPU load.
    5_000,  # shoulder  final: 2_000
    5_000,  # elbow  final: 2_500
    1_000,  # vertical  final: 1_000
]

SHOULDER_LENGTH = 18.45  # 24.5
ARM_LENGTH = 31
FOREARM_LENGTH = 32.25

ORIGIN_OFFSET = (16.5, -6.5)
BOARD2FRAME_DISTANCE = (7, 36)

VERTICAL_TOTAL_STEP_HEIGHT = 0


def wait_ns(delay: int) -> int:
    """
    Wait for the specified number of nanoseconds.
    Returns the number of nanoseconds that were actually waited.

    Precision may vary depending on CPU load, switch interval or other factors.

    If delay is lower than 10 times the system's switch interval, the CPU is not
    yielded to other threads.
    """
    # start time must be obtained as soon as this function begins to take into
    # account the overhead of checking the system's switch interval.
    # Additional overhead does not matter for time scales similar to the system's
    # switch interval.
    start_time = time.time_ns()

    # if delay is higher than 10 times the switch interval, fallback to `time.sleep`, which does yield the
    # CPU to other threads.
    if delay > (sys.getswitchinterval() * 1e10):
        time.sleep(delay / 1e9)
        return delay

    # otherwise, a while loop will do the trick.
    while (time.time_ns() - start_time) < delay:
        pass
    return time.time_ns() - start_time


def wait_us(delay: int) -> float:
    """
    Wait for the specified amount of microseconds.
    Returns the actual number of microseconds waited as a float.

    Precision may vary depending on CPU load, switch interval or other factors.

    If delay is lower than 10 times the system's switch interval, the CPU is not
    yielded to other threads.
    """
    return wait_ns(delay * 1000) / 1000


class Axle(IntEnum):
    SHOULDER = 0
    ELBOW = 1
    VERTICAL = 2


class Direction(IntEnum):
    COUNTER_CLOCKWISE = 0
    CLOCKWISE = 1


class RoboticArm:
    def __init__(self, ardu_com_port, calibrate=False):
        self._axle_positions = [0, 0, 0]  # counted in steps
        self._bounds = [
            (1, 1),  # shoulder
            (0, 1),  # elbow
            (0, 1),  # vertical
        ]

        # board
        self._board = pyfirmata.ArduinoMega(ardu_com_port)
        start_iterator(self._board)

        # pins
        self._step_pins = (
            self._board.get_pin('d:41:o'),  # shoulder
            self._board.get_pin('d:51:o'),  # elbow
            self._board.get_pin('d:31:o'),  # vertical
        )
        self._dir_pins = (
            self._board.get_pin('d:40:o'),  # shoulder
            self._board.get_pin('d:50:o'),  # elbow
            self._board.get_pin('d:30:o'),  # vertical
        )
        self._stop_pins = (
            (self._board.get_pin('d:5:i'), self._board.get_pin('d:4:i')),  # shoulder: (clockwise, counterclockwise)
            (self._board.get_pin('d:3:i'), self._board.get_pin('d:2:i')),  # elbow: (clockwise, counterclockwise)
            (self._board.get_pin('d:7:i'), self._board.get_pin('d:6:i')),  # vertical (bottom, top)
        )

        # self._led_pin = self._board.get_pin('d:11:o')

        if calibrate:
            self.calibrate()

        self._chain: Chain | None = None

    def _create_chain(self):
        self._chain = Chain(
            [
                OriginLink(),
                URDFLink(
                    'shoulder',
                    [SHOULDER_LENGTH, 0, 0],
                    [0, 0, 0],
                    bounds=self._degrees_bounds(Axle.SHOULDER),
                    rotation=[0, 0, 1],
                ),
                URDFLink(
                    'elbow',
                    [ARM_LENGTH, 0, 0],
                    [0, 0, 0],
                    bounds=self._degrees_bounds(Axle.ELBOW),
                    rotation=[0, 0, 1],
                ),
                URDFLink(
                    'hand',
                    [FOREARM_LENGTH, 0, 0],
                    [0, 0, 0],
                    rotation=[0, 0, 0],
                    # joint_type="fixed",
                ),
            ],
            active_links_mask=[False, True, True, False]
        )

    def _degrees_bounds(self, axle):
        lower, upper = self._bounds[axle]
        return (
            lower * AXLE_STEP_ANGLES[axle],
            upper * AXLE_STEP_ANGLES[axle],
        )

    def _degrees_axle_position(self, axle):
        return round(self._axle_positions[axle] * AXLE_STEP_ANGLES[axle])

    def _calibrate(self, axle):

        # find the first end of track sensor
        while not self.get_end_of_track(axle, Direction.CLOCKWISE):
            self._rotate_step(axle, Direction.CLOCKWISE)

        steps = 0
        time.sleep(1)

        # find the second end of track sensor and measure the number of steps
        while not self.get_end_of_track(axle, Direction.COUNTER_CLOCKWISE):
            self._rotate_step(axle, Direction.COUNTER_CLOCKWISE)
            steps += 1

        secondhalf_steps = steps // 2
        firsthalf_steps = steps - secondhalf_steps

        time.sleep(1)

        # place ourselves in the middle
        for _ in range(steps // 2):
            self._rotate_step(axle, Direction.CLOCKWISE)

        if self._bounds[axle][0]:
            self._axle_positions[axle] = 0
            self._bounds[axle] = (
                -firsthalf_steps,
                secondhalf_steps,
            )
        else:
            self._axle_positions[axle] = firsthalf_steps
            self._bounds[axle] = (
                0,
                steps,
            )

    def calibrate(self):
        for axle in (Axle.VERTICAL, Axle.SHOULDER, Axle.ELBOW):
            self._calibrate(axle)
            print(f"{axle}: {self._bounds[axle]}")

        self._create_chain()

    def _rotate_step(self, axle, direction):
        delay = AXLE_DELAYS[axle]
        self._dir_pins[axle].write(not direction)

        self._step_pins[axle].write(1)
        wait_us(delay)
        self._step_pins[axle].write(0)
        wait_us(delay)

    @staticmethod
    def _square2coords(i, j):
        if (i is None) and (j is None):
            i = -2
            j = 0

        return (
            5*i + BOARD2FRAME_DISTANCE[0] - ORIGIN_OFFSET[0] + 47.5,
            5*j + BOARD2FRAME_DISTANCE[1] + ORIGIN_OFFSET[1] + 7.5,
        )

    def get_end_of_track(self, axle, direction):
        return not self._stop_pins[axle][direction].read()

    def _inverse_kinematics(self, x, y):
        shoulder_angle = self._degrees_axle_position(Axle.SHOULDER)
        elbow_angle = self._degrees_axle_position(Axle.ELBOW)


        target_vector = [x, y, 0]
        target_frame = numpy.eye(4)
        target_frame[:3, 3] = target_vector

        print(f"before ik, current values: {shoulder_angle=} {elbow_angle=}")
        print(f"target position: {x=} {y=}")

        angles = self._chain.inverse_kinematics(
            target_vector,
            initial_position=[0, shoulder_angle, elbow_angle, 0],
        )
        print(f"{angles=}")
        return angles[1:3]

    def rotate_multiple(self, shoulder=0, elbow=0):  # angles are in degrees

        shoulder += (self._axle_positions[Axle.SHOULDER] * AXLE_STEP_ANGLES[Axle.SHOULDER])
        elbow += (self._axle_positions[Axle.ELBOW] * AXLE_STEP_ANGLES[Axle.ELBOW])

        return self.set_angles(shoulder, elbow)


    def set_angles(self, shoulder, elbow, steps=False):  # angles are in degrees if steps is False and in steps otherwise
        if not steps:
            # convert the angle into steps
            steps_shoulder = int(shoulder // AXLE_STEP_ANGLES[Axle.SHOULDER])
            steps_elbow = int(elbow // AXLE_STEP_ANGLES[Axle.ELBOW])

            print("steps", 'elbow:', steps_elbow, "shoulder:", steps_shoulder)
            print(f"{self._axle_positions=}")

            return self.set_angles(steps_shoulder, steps_elbow, True)

        goal = (
            shoulder,
            elbow,
        )
        done = [False, False]
        directions = [None, None]
        signs = [None, None]

        # compute specific values
        for axle in (Axle.SHOULDER, Axle.ELBOW):
            diff = goal[axle] - self._axle_positions[axle]
            directions[axle] = Direction.CLOCKWISE if diff > 0 else Direction.COUNTER_CLOCKWISE
            signs[axle] = abs(diff) // diff if diff else 0  # abs(x) / x returns -1 for negative numbers and 1 for positive numbers


        while not all(done):

            for axle in (Axle.SHOULDER, Axle.ELBOW):
                # if the motor has reached its bounds or its goal, it is done
                if (
                    done[axle] or  # avoid the other computations if we already know it is done
                    (self._axle_positions[axle] == goal[axle]) or
                    (self._axle_positions[axle] in self._bounds[axle])
                ):
                    done[axle] = True
                    continue

                # move the motor by one in the right direction and update our current position
                self._rotate_step(axle, directions[axle])
                self._axle_positions[axle] += signs[axle]

    def set_position(self, x, y):

        shoulder_angle, elbow_angle = self._inverse_kinematics(x, y)

        """print(f"before {shoulder_angle=}, {elbow_angle=}")

        shoulder_angle += self._degrees_axle_position(Axle.SHOULDER)
        elbow_angle += self._degrees_axle_position(Axle.ELBOW)"""

        print(f"after {shoulder_angle=}, {elbow_angle=}")

        return self.set_angles(shoulder_angle, elbow_angle)

    def reach_square(self, i, j):
        return self.set_position(*self._square2coords(i, j))

    def grab_piece(self, piece_type):
        pass

    def release_piece(self):
        pass

    def move_piece(self, src, dst, piece_type):
        self.reach_square(*src)
        self.grab_piece(piece_type)
        self.reach_square(*dst)
        self.release_piece()

    def make_move(self, src, dst, src_piece, dst_piece, en_passant_square=None, castling_side=None):
        if dst_piece is not None:  # if there already is a piece at the destination, remove it first
            self.move_piece(dst, (None, None), dst_piece.piece_type)  # captured piece goes outside the board

        if en_passant_square is not None:  # for en passant moves the captured piece is not where the move destination is
            self.move_piece(en_passant_square, (None, None), chess.PAWN)

        if castling_side is not None:

            # branchless programming techniques:
            cond = bool(castling_side)
            rook_src_file = cond * 7  # 7 if cond else 0
            rook_dst_file = 2 + (cond * 2)  # 2+2 if cond else 2
            king_dst_file = 1 + (cond * 4)  # 1+4 if cond else 1
            king_src_file = 5

            self.move_piece((king_src_file, 0), (king_dst_file, 0), chess.KING)
            self.move_piece((rook_src_file, 0), (rook_dst_file, 0), chess.ROOK)

        self.move_piece(src, dst, src_piece.piece_type)  # then perform the corresponding move

    def rest_position(self):
        self.set_angles(90, 90)

    def plot(self):
        print(f"{self._degrees_bounds(Axle.SHOULDER)=}")

        ax = pyplot.figure().add_subplot(111, projection='3d')
        k = self._chain.inverse_kinematics([1, 1, 0])
        print(k)
        self._chain.plot(k, ax)  # [0, 0, 0]


def test():
    arm = RoboticArm('COM3')

    print('Press enter when alim is connected.')
    input()

    arm.calibrate()
    arm.rest_position()

