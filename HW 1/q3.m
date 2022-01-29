syms x w1 b1 w2 b2;
f1 = x*w1 + b1;
a = 1/(1+exp(-f1));
f2 = a*w2+b2
jacobian(f2,x)