#!/usr/bin/env python3
"""Minimal servo test - just open and close 3 times"""

import time
import RPi.GPIO as GPIO

SERVO_PIN = 18

GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, 50)
servo.start(5.5)  # Open

try:
    for i in range(3):
        print(f"Cycle {i+1}")
        servo.ChangeDutyCycle(5.5)   # Open
        time.sleep(1)
        servo.ChangeDutyCycle(9.0)   # Clamp
        time.sleep(1)
        servo.ChangeDutyCycle(5.5)   # Open
        time.sleep(1)
finally:
    servo.stop()
    GPIO.cleanup()
    print("Done!")