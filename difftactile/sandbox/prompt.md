I'm using the DiffTactile fully differentiable physics simulator to simulate a soft optical tactile sensor (ViTacTip) that collides with a silicone phantom that mimics human tissue. I model the sensor and the phantom as using a single material each. The simulator uses MLS-MPM (based on github.com/yuanming-hu/taichi_mpm/blob/master/mls-mpm88-explained.cpp), the neo-hookean elastic model and Corotated Linear Elastic model for the phantom, and FEM for the sensor. The code uses SI units throughout.

I have an issue. When I set the Young's modulus of the sensor and of the phantom to have equal values, the sensor doesn't deform during a downward press into the phantom. In order for the sensor to deform, I need to set its Young's modulus value to be 1 order of magnitude less than the Young's modulus value of the phantom. In reality, the sensor is stiffer than the phantom and both the sensor and the phantom deform during the collision.

Please review my physics code and try to identify any potential culprits.

