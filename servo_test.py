import RPi.GPIO as GPIO # GPIO Library
# sudo apt-get update && sudo apt-get install python3-rpi.gpi
from time import sleep

GPIO.setmode(GPIO.BOARD) # access pins
GPIO.setup(18, GPIO.OUT) # connect to pin 18

pwm = GPIO.PWM(18, 50) # set up 50Hz PWM on pin 18
pwm.start(0)

# Test motor (duty cycle / period)
pwm.ChangeDutyCycle(5) # left
sleep(1)
pwm.ChangeDutyCycle(7.5) # neutral
sleep(1)
pwm.ChangeDutyCycle(10) # right
sleep(1)

# exit
pwm.stop()
GPIO.cleanup()
