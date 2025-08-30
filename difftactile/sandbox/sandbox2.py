# Inputs per contact:
# m_n: mass of MPM node
# v_n: nodal velocity vector
# tri: triangle with vertex masses m_s1,m_s2,m_s3 and velocities v_s1,v_s2,v_s3
# contact_point: barycentric weights alpha = [a1,a2,a3]
# n: contact normal (pointing from sensor into phantom)
# e: restitution coefficient (0..1)
# dt: timestep

# compute contact-point velocity (point = sum(alpha_i * v_si))
v_s_point = a1*v_s1 + a2*v_s2 + a3*v_s3
v_r = dot(v_n, n) - dot(v_s_point, n)
if v_r >= 0:
    # separating or just touching, no normal impulse needed
    continue

# denominator = 1/m_n + sum(alpha_i^2 / m_si)
den = 1.0/m_n + (a1*a1)/m_s1 + (a2*a2)/m_s2 + (a3*a3)/m_s3

# impulse scalar along normal
J = - (1.0 + e) * v_r / den

# apply updates (velocity impulse form)
v_n += (J / m_n) * n
v_s1 -= (a1 * J / m_s1) * n
v_s2 -= (a2 * J / m_s2) * n
v_s3 -= (a3 * J / m_s3) * n

# (If you store momenta directly, add/subtract J*n to momenta)
