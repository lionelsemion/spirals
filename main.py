import numpy as np
import scipy.signal as sig
import cv2
from math import tau, sin, cos
from random import randrange, random


granularity = 4

circle = np.zeros((1 + 2 * granularity, 1 + 2 * granularity), np.float32)
circle = cv2.circle(circle, (granularity, granularity), granularity, 1.0)

input_image = np.zeros((512, 512), np.float32)
input_image = cv2.circle(input_image, (256, 256), 128 + 64, 1.0, -1)
input_image = cv2.circle(input_image, (256 + 64, 256 - 64), 128 + 32, 0.0, -1)

shape = input_image.shape

output_image = np.uint8(np.ones((*shape, 3), np.uint8) * (0, 0, 0))

accessible = 1 - sig.convolve2d(1 - input_image, circle, mode="same")

accessible = cv2.rectangle(
    accessible, (0, 0), (shape[1] - 1, shape[0] - 1), 0.0, 3 + granularity
)


def turn_till_free(accessible, x, y, direction, step_size=granularity, inverse=False):
    turn_rate = tau / 64
    direction -= tau / 2
    turned = 0
    while 1:
        try:
            if (
                accessible[int(y + sin(direction) * step_size * 1.0)][
                    int(x + cos(direction) * step_size * 1.0)
                ]
                > 0.1
            ):
                return direction + random() * tau / 32 * (1 if inverse else -1)
        except IndexError:
            pass
        turned += turn_rate
        direction += turn_rate if inverse else -turn_rate
        if turned > 2 * tau:
            return None


i = 0

while len(next_points := np.argwhere(accessible > 0.2)) > 0:
    y, x = next_points[randrange(len(next_points))]
    direction = random() * tau
    # output_image = cv2.circle(output_image, (int(x), int(y)), 5, (0, 0, 255), -1)
    inverse = random() < 0.5
    j = 0
    while 1:
        direction = turn_till_free(accessible, x, y, direction, granularity, inverse)

        old_point = (x, y)

        accessible = cv2.circle(
            accessible, (int(old_point[0]), int(old_point[1])), granularity, 0.0, -1
        )

        if direction is None:
            break

        if (j := j + 1) > 100:
            break

        x, y = x + cos(direction) * granularity, y + sin(direction) * granularity

        output_image = cv2.line(
            output_image,
            (int(old_point[0]), int(old_point[1])),
            (int(x), int(y)),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if (i := i + 1) % 1 == 0:
            # cv2.imshow("Input Image", input_image)
            # cv2.imshow("Accessible Space", accessible)
            cv2.imshow("Output Image", output_image)
            cv2.waitKey(1)

cv2.imshow("Output Image", output_image)
cv2.waitKey()


# MUSIC
