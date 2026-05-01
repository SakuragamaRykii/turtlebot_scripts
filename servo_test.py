import time
import pigpio

SERVO_GPIO = 18

SAFE_MIN = 1000
SAFE_MID = 1500
SAFE_MAX = 2000

pi = pigpio.pi()
if not pi.connected:
    raise RuntimeError("pigpio daemon not connected")

def set_pulse(us):
    us = max(SAFE_MIN, min(SAFE_MAX, us))
    pi.set_servo_pulsewidth(SERVO_GPIO, us)
    print(f"pulse = {us}")

try:
    print("Move to middle")
    set_pulse(SAFE_MID)
    time.sleep(1.5)

    print("Move to one side")
    set_pulse(SAFE_MIN)
    time.sleep(1.5)

    print("Move to other side")
    set_pulse(SAFE_MAX)
    time.sleep(1.5)

    print("Back to middle")
    set_pulse(SAFE_MID)
    time.sleep(1.5)

finally:
    pi.set_servo_pulsewidth(SERVO_GPIO, 0)
    pi.stop()