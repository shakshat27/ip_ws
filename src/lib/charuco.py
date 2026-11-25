"""The ChArUco calibration board, defined once from cell.yaml calibration.charuco."""
import cv2


def board_from(spec):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec["dictionary"]))
    return cv2.aruco.CharucoBoard_create(spec["squares"][0], spec["squares"][1], spec["square"], spec["marker"], d), d
