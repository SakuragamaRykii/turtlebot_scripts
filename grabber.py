import RPi.GPIO as GPIO

from time import sleep

pwm = None

def setup_motor():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(18, GPIO.OUT)

    pwm = GPIO.PWM(18, 50)
    pwm.start(0)

    pwm.ChangeDutyCycle(10)

def set_grabber(closed: bool):
    if (closed):
        pwm.ChangeDutyCycle(5)
    else
        pwm.ChangeDutyCycle(10)

set_grabber(True)
sleep(3)
set_grabber(False)
sleep(1)
set_grabber(True)

pwm.stop()
GPIO.cleanup()

'''
def set_angle(angle):
    duty = angle / 18 + 3
    # GPIO.output(11, True)
    pwm.ChangeDutyCycle(duty)
    # sleep(1)
    # GPIO.output(11, False)
    # pwm.ChangeDutyCycle(duty)
'''
