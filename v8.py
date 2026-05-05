#!/usr/bin/env python3
import math
import time

import rclpy

import v7 as base


SEARCH_TRACK_MAX_ANG = 0.16
SEARCH_LOCK_PIXEL = 170
SEARCH_EDGE_PIXEL = 260
ALIGN_LOST_TIMEOUT = 1.25
ALIGN_RECOVER_ANG = 0.08
V8_CONFIRM_FRAMES = 3


class TrackingCubeV8(base.TrackingCubeV7):
    def __init__(self):
        super().__init__()
        self.candidate_color = None
        print('[BOOT] TrackingCube v8 patch: stable cube lock + safer loss recovery')

    def cube_recently_seen(self):
        return (time.monotonic() - self.last_target_seen_time) <= ALIGN_LOST_TIMEOUT

    def cube_confirmed(self):
        if not self.target_visible or self.target_obs is None:
            return False
        if self.target_obs.missing > 2:
            return False
        if self.target_seen_frames < V8_CONFIRM_FRAMES:
            return False
        err = self.cube_error_pixels()
        if err is None:
            return False
        return abs(err) <= SEARCH_LOCK_PIXEL

    def clear_ungrabbed_plan(self):
        self.candidate_color = None
        if self.state in ('SEARCH_CUBE', 'ALIGN_CUBE', 'APPROACH_CUBE'):
            self.carried_color = None
            self.destination_marker_id = None

    def prepare_delivery_plan(self, color):
        self.candidate_color = color
        self.carried_color = color
        self.destination_marker_id = self.destination_for_color(color)

    def handle_search_cube(self):
        if self.front_dist <= base.EMERGENCY_STOP_M:
            self.set_state('STOPPED', 'emergency front distance')
            return

        if self.target_visible and self.target_obs is not None and self.image_width is not None:
            err = self.cube_error_pixels()

            if self.cube_confirmed():
                self.candidate_color = self.target_obs.color
                self.stop_robot_once()
                self.set_state('ALIGN_CUBE', f'locked {self.candidate_color}')
                return

            if err is not None and abs(err) <= SEARCH_EDGE_PIXEL and self.target_obs.missing <= 4:
                err_norm = err / max(self.image_width / 2.0, 1.0)
                angular = self.clamp_abs(-0.22 * err_norm, base.ALIGN_MIN_ANG, SEARCH_TRACK_MAX_ANG)
                self.publish_cmd(0.0, angular)
                return

        self.clear_ungrabbed_plan()
        self.publish_cmd(0.0, base.SEARCH_ANG * 0.72)

    def handle_align_cube(self):
        if self.front_dist <= base.EMERGENCY_STOP_M:
            self.set_state('STOPPED', 'emergency front distance')
            return

        if not self.target_visible or self.target_obs is None:
            if self.lost_cube_close_should_grab() and self.candidate_color is not None:
                self.prepare_delivery_plan(self.candidate_color)
                self.set_state('GRAB_FORWARD', 'cube lost close, continue grab')
                return
            if self.cube_recently_seen():
                self.publish_cmd(0.0, self.last_target_dir * ALIGN_RECOVER_ANG)
                return
            self.clear_ungrabbed_plan()
            self.set_state('SEARCH_CUBE', 'cube lost before grab')
            return

        self.candidate_color = self.target_obs.color
        err = self.cube_error_pixels()
        if err is None:
            self.stop_robot_once()
            return

        if abs(err) <= base.CUBE_ALIGN_PIXEL_TOL:
            self.centered_frames += 1
            self.stop_robot_once()
            if self.centered_frames >= base.CENTER_HOLD_FRAMES:
                self.set_state('APPROACH_CUBE', f'centered {self.target_obs.color}')
            return

        self.centered_frames = 0
        err_norm = err / max(self.image_width / 2.0, 1.0)
        angular = self.clamp_abs(-0.22 * err_norm, base.ALIGN_MIN_ANG, base.ALIGN_MAX_ANG * 0.82)
        self.publish_cmd(0.0, angular)

    def handle_approach_cube(self):
        if self.front_dist <= base.EMERGENCY_STOP_M:
            self.set_state('STOPPED', 'emergency front distance')
            return

        if not self.target_visible or self.target_obs is None:
            if self.lost_cube_close_should_grab() and self.candidate_color is not None:
                self.prepare_delivery_plan(self.candidate_color)
                self.set_state('GRAB_FORWARD', 'cube disappeared close, forward before servo')
                return
            if self.cube_recently_seen():
                self.stop_robot_once()
                return
            self.clear_ungrabbed_plan()
            self.set_state('SEARCH_CUBE', 'cube lost far away')
            return

        self.candidate_color = self.target_obs.color

        if self.cube_close_enough_to_grab():
            self.prepare_delivery_plan(self.target_obs.color)
            self.set_state('GRAB_FORWARD', 'cube close enough')
            return

        err = self.cube_error_pixels()
        if err is None:
            self.stop_robot_once()
            return

        if abs(err) > 130:
            self.stop_robot_once()
            self.set_state('ALIGN_CUBE', 'cube off center')
            return

        err_norm = err / max(self.image_width / 2.0, 1.0)
        linear = base.APPROACH_SLOW_SPEED if self.front_dist <= 0.18 or self.target_obs.bbox_h >= 160 else base.APPROACH_FAST_SPEED
        angular = 0.0 if abs(err) <= base.CUBE_ALIGN_PIXEL_TOL else self.clamp_abs(-0.18 * err_norm, 0.02 * base.SPEED_SCALE, 0.08 * base.SPEED_SCALE)
        self.publish_cmd(linear, angular)

    def handle_servo_down(self):
        if self.destination_marker_id is None:
            color = self.candidate_color or self.carried_color or 'blue'
            self.prepare_delivery_plan(color)
        self.stop_robot_reliable(repeat=5, delay=0.02)
        self.servo.servo_down()
        self.start_turn_relative(math.pi, 'TURN_TO_ZONE')


def main(args=None):
    rclpy.init(args=args)
    node = TrackingCubeV8()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.request_shutdown('KeyboardInterrupt')
        node.stop_robot_reliable()
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
