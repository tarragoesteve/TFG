The polynomial layer it's a fuction that revice as an input N variables an output the sum of all monomials up to degree D multiplied by a coeficient. This coeficient is a trainable variable.

For example with two variables x_1 and x_2 and degree 2 we obtain the following polynomial: a_0 + a_1 \times x_1 + a_2 \times x_2 + a_3 \times x_1^2 + a_4 \times x_2^2 + a_5 \times x_1 \times x_2

The number of variables of the layer is equal to the number of monomials up to degree D of N variables that can be expressed with the following formula:

#variables = ![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/5b2d96856677a9b9f5d9f3d67b52b0d5be22f8f7)

Some results:

N=25 D=2: 351

N=25 D=3: 3276

N=800 D=2: 321,201

N=800 D=3: 85,974,801

For more information on monomials: https://en.wikipedia.org/wiki/Monomial

This is the reason why we will start focusing on small number of inputs as filters and see if higher polynomial degree benefits the overall results.
