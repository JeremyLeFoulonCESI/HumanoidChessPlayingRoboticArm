import numpy

import application
import sys

from ikpy.chain import Chain
from ikpy.link import URDFLink, OriginLink
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

import robot


if __name__ == '__main__':

    if not ((3, 10) <= sys.version_info < (3, 11)):
        raise NotImplementedError("Sorry, this program only works in python version 3.10.X.")

    """chain = Chain(
            [
                OriginLink(),
                URDFLink(
                    'shoulder',
                    [18.45, 0, 0],
                    [0, 0, 0],
                    bounds=(-90, 90),
                    rotation=[0, 0, 1],
                ),
                URDFLink(
                    'elbow',
                    [31, 0, 0],
                    [0, 0, 0],
                    bounds=(-90, 90),
                    rotation=[0, 0, 1],
                ),
                URDFLink(
                    'wrist',
                    [32.25, 0, 0],
                    [0, 0, 0],
                    rotation=[0, 0, 0],
                ),
            ],
            active_links_mask=[False, True, True, True]
        )


    ax = pyplot.figure().add_subplot(111, projection='3d')
    chain.plot(chain.inverse_kinematics([45, 55, 0]), ax)
    pyplot.show()"""

    # robot.test()

    application.Application.run()

