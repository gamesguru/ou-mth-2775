***************************
 Linear Algebra, edition 9
***************************

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sample Exam II (questions)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- 4(b) negative signs on span/basis (first element)
- 8(a)(b) [see Section 3.6 questions]



Review (self - Final)
#####################

Fundamental Subspaces Theorem and # solutions (to Ax=b problems)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Theorem 5.2.1: If ``A`` is ``m x n``, then **N(A) = R(Aᵀ)** :sup:`⊥` and **N(Aᵀ) = R(A)** :sup:`⊥`.

Proof:

We already know ``N(A) ⊥ R(Aᵀ)`` and, therefore, ``N(A) ⊂ R(Aᵀ)`` :sup:`⊥`.

On the other hand, if **x** is any vector in ``R(Aᵀ)`` :sup:`⊥`, then **x** is orthogonal to each column vector in ``Aᵀ``, so ``Ax = 0``.

Thus, ``x`` must be an element of ``N(A)``. So, ``N(A) = R(Aᵀ)`` :sup:`⊥`.

This also holds for square matrices. If ``B=Aᵀ``, then

N(Aᵀ) = N(B) = R(Bᵀ) :sup:`⊥` = R(A) :sup:`⊥`.

QED




~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Chapter 2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


2.2 - Properties of Determinants
################################

Why is it that adding **columns** to each other also doesn't change the value of the determinant?
Can you do both row and column operations? Or can you only pick one or the other?



2.3 - Auxiliary topics (Optional)
#################################

How is any of that stuff true?




~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Chapter 3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


3.2 - Subspaces
###############

You *can* do the ``a*x ∈ S`` and ``x + y ∈ S`` steps separately, but you can also do ``a*x + b*y ∈ S`` in one step.

**NOTE:** ``{0} and V`` are subspaces of ``V``, but *only other* subspaces are called **proper subspaces**. ``S``is non-empty since ``0 ∈ S`` is guaranteed.


Any challenging HW problems?

- Example 6 (nth derivative is a subspace of all functions)
- Example 11 (determining if a set of vectors spans a space by use of matrix solutions)


Linear Systems Revisited
~~~~~~~~~~~~~~~~~~~~~~~~

Why if a *unique* ``x0`` solution exists to ``Ax=b`` is *one* vector from the null space together with the ``x0`` vector span the solution space?

**Come back to this!**



3.3 - Linear Independence
#########################

PT #1
~~~~~

``v1=(1,2,3), v2=(2,-1,0), v3=(-2,3,-1), v4=(-1,2,3)``

Attempt 1

``v1 - v4 = (2,0,0)``

``v2 - (v1 - v4) = (0,-1,0)``

``v3 + (v1 - v4) - 3[v2 - (v1 - v4)] = (0,0,-1)``


Reading 3.3
~~~~~~~~~~~

If any vector is a sum of multiples of the others, the system is not linearly independent.

If any vector is a multiple of another, one of them can be ruled out from the Basis (span).

If two vectors in ``R^2`` are L.I., they are scalar multiples of each other.

Same with ``R^3``: unless ``v1``, ``v2``, and the origin ``(0,0,0)`` are collinear... the two vectors are **not** L.I.

- Any vector ``v3`` is only linearly dependent if it exists on the plane (or line) which ``span{v1, v2}``. Otherwise, it's out of their span and is L.I.
- ``n`` vectors are L.D. ``iff`` their ``X`` matrix is singular
- You can use the determinant. if ``det(X) = 0``, the vectors are L.D.
- For **non-square** systems:

  + Only if there are **free vars** is it L.D.
  + Otherwise, there are no free vars, and ``c=0`` is the only solution

If L.I., every representation of ``v_n`` in terms of the other ``n-1`` vectors is **unique**.
If L.D., there is at least one other combination.


Example 6

There are some polynomial vectors which are linearly dependent.

Example 7

To show ``e^x`` and ``e^-x`` are L.I., you can use the **Wronskian** matrix!

- If it's zero, they're linearly dependent.
- Otherwise, **we don't know** and need to check!

Example 8

``f=x^2, g=x*|x|``

Only solution is ``c1 = c2 = 0``, so the system is L.I.

Example 9

``1, x, x^2, x^3`` are L.I. in ``C(-inf, inf)``



3.3 - Basis & Dimension
#######################

A spanning set is minimal if its elements are L.I.

The elements of a minimum spanning set form a **basis** of that vector space ``V``.

Def: ``v1, v2, ..., v_n`` form a **basis** of V ``iff``

- vectors are linearly independent,
- vectors span V.

There are many **non-standard** bases besides ``{e1, e2, e2}``.
Any basis for ``R^3`` must have exactly 3 vectors.

Example 2

``R^2x2`` is made of four L.I. elementary matrices ``E11, E12, E21, E22``.

Finding the **null space** of matrix gives you solutions to the homogeneous equation.
You can find the basis and dimension based on decomposing the solutions into sums of multiples of the bases vectors.

Theorem 3.4.1 - any collection of ``n+1`` vectors (in a space ``V`` spanned by ``n`` vectors) are L.D.
(see proof on page 158)

All **bases** of the same spanning set have the **same dimension**.

Possible dimensions:

- ``n`` vectors -> ``n``
- ``{0}`` -> 0
- ``{1,1,0}`` -> 2
- ``{1,0,0}`` -> 1


Example 3 - Prove that vector space f ``P`` (polynomials) is infinite dimensional. (Brain teaser)

Example 4 - Show that ``(1,2,3)T, (-2,1,0)T, and (1,0,1)T`` are a basis for ``R^3``.

We need only show that these three vectors are L.I., which we can easily accomplish with ``det(v1|v2|v3) = 2``.

**NOTE:** for non-standard bases, see **least squares problem** in Ch 5 or **eigenvalue** applications in Ch 6.



3.5 - Change of Basis
#####################

**TODO:** this.



3.6 - Row space & Column space
##############################

Pre-reading ideas (3.6)
~~~~~~~~~~~~~~~~~~~~~~~

**Question:** Why does the column size (or minimum size) matter most in determining the number of solutions?

- 3.6 #7, 9


Good video on the fundamental theorem of linear algebra

The Four Fundamental Subspaces and the Fundamental Theorem | Linear Algebra - YouTube
https://www.youtube.com/watch?v=eeGvVyesafw


Why are these three cases true?

linear algebra - number of solutions and rank - Mathematics Stack Exchange
https://math.stackexchange.com/questions/752941/number-of-solutions-and-rank


Reading - 3.6
~~~~~~~~~~~~~

Def: ``A (m x n matrix)``

- Row space = subspace of ``R^(1 x n)`` spanned by rows
- Col space = subspace of ``R^m`` spanned by columns

Theorem 3.6.1 - Two row equivalent matrices have the same row space.

Def: **rank** is the dimension of the row space

Theorem 3.6.2 - ``Ax = b`` is consistent <=> ``b ∈ C(A)`` (b in col space of A)

- ``Ax = 0`` has trivial solution ``x=0`` iff col vectors of A are L.I.

Theorem 3.6.2 - ``Ax = b`` is consistent for every ``b ∈ R^m`` iff col vectors span ``R^m``

- ``Ax = b`` has at most one solution for every ``b ∈ R^m`` iff the col vectors of A are L.I.

**NOTE:** if col vectors span ``R^m``, then ``n>=m`` (at least as many rows as columns).

Corollary 3.6.4 - ``n x n`` square matrix ``A`` is non-singular iff col vectors of ``A`` form a basis for ``R^n``.

Theorem 3.6.5 - Let ``A`` be an  ``(m x n)`` matrix, then ``rank(A) + nullity(A) = n``

Theorem 3.6.6 - ``dim(R(A)) = dim(C(A))`` (see proof on page 176)


**NOTE:** In ``U = rref(A)``, the leading entries in ``U`` determine which columns to choose from ``A`` to span ``C(A)``. (in general ``C(A) != C(U)``)

(see Example 4)

Example 5 - subspace spanned in ``R^4`` by four vector needn't have ``dim 4``. Two leading entries => two columns span ``C(A)``.




~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Chapter 4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

4.1 - Linear Transformations
############################

Def: ``L(a*v1 + b*v2) = a*L(v1) + b*L(v2)`` true for all ``a,b ∈ R``, and ``v1,v2 ∈ V``

Notation: ``L: V -> W`` (NOTE: if ``V = W``, then ``L`` is a **linear operator** over ``V``)

Example 2 - ``L(x) = x1 * e1`` is a L.O. for every ``x ∈ R^2``

Example 3 - so is ``L(x) = (x1, -x2)T``

Example 5 - ``L(x) = x1 + x2`` is a linear mapping ``L: R^2 -> R``

Example 6 - ``M(x) = Sqrt(x1^2 + x2^2)`` is not, as scalar multiplication is **not** closed over ``M``

Example 7 - try it yourself (page 189)


We can define a matrix s.t. ``L(x) = Ax`` for every ``x ∈ R^n``, where ``L: R^n -> R^m``

- ``L(O_v) = 0_w``
- ``L(a1*v1 + ... + a_n*v_n) = a1 * L(v1) + ... + a_n * L(v_n)``
- ``L(-v) = -L(v)``

Example 8 - Identity operator is ``I(v) = v)``, a L.O.

Example 9 - Definite **integral** mapping ``L: C[a,b] -> R`` is a L.T.

Example 10 - So is the derivative operator


Def: ``L: V-> W``, then ``Ker L = {v ∈ V | L(v) = 0_w}``

Def: ``L: V->W`` and ``S`` is subspace of ``V``. The **image** ``L(S) = {w ∈ W | w = L(v) for some v ∈ S}``

- The image of the *entire* vector space, ``L(V)``, is called the **range** of L

Theorem 4.1.1 - ``Ker(L)`` is a subspace of ``V``, and ``Range(S)`` is a subspace of ``W``


Example 11 - Let ``L(x) = (x1, 0)T``. Then ``x ∈ ker(L)`` iff ``x1=0``, so the **kernel** is the one-dimension subspace spanned by ``e2=(0, 1)``. The **range** is the space spanned by ``e1``.

Example 12 - Let ``L: R^3 -> R^2`` and ``L(x) = (x1 + x2, x2 + x3)T``.

- ``x ∈ ker(L) <=> x1 + x2 = 0 and x2 + x3 = 0``

  + set ``x3 = a``, the free var... then ``x2 = -a`` and ``x1 = a``.
  + Then ``ker(L) = span{(1,-1,1)T}``

- ``x ∈ S => x = (a,0,b)T``

  + Hence, ``L(x) = (a,b)T``.
  + So, range: ``L(R^3) = R^2``.


Example 13 - ``D: P_3 -> P_3`` differentiation operator

- ``ker(D) = 0`` (zero degree polynomials)
- Range: ``D(P_3) = P_2``



4.2 - Matrix Representation of L.T.
###################################

Theorem 4.2.1 ``L: R^n -> R^m``, there exists a matrix ``L(x) = Ax`` where ``A`` is an ``m x n`` matrix.

(proof: see page 195)

**Review:** Examples 4, 5, and 6




~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Chapter 5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


5.1 - Scalar Product
####################

Def: ``xT*y = x1*y1 + ... + x_n*y_n``

Distance from x to y: ``|x - y|``

Theorem 5.1.1 ``xT*y = |x| |y| cos(theta)`` (for ``R^2`` and ``R^3``)

Cauchy-Schwarz Inequality: ``|xT*y| <= |x| |y|`` (for ``R^2`` and ``R^3``)


Def: x and y are orthogonal if ``xT*y=0`` (for ``R^2`` and ``R^3``)

**Review:** What are the **scalar projection** and **vector projection**?

**NOTE:** The above theorems can be generalized to ``R^n``.



5.2 - Orthogonal Subspaces
##########################

Def: Two subspaces ``X, Y`` are **orthogonal** if ``xT*y=0`` for every ``x ∈ X and y ∈ Y``.

Def: **Orthogonal complement**  of ``Y`` is ``Yp = {x ∈ R^n | xT*y=0 for every y ∈ Y}``

- ``X intersect Y = {0}``
- ``Y subspace of R^n => Yp also subspace of R^n``


From chapter 3, ``b ∈ R^m`` is in ``C(A)`` iff ``Ax = b`` for some ``x ∈ R^n``.

- ``C(A) = range(A)``
- ``Range(A)   = {b ∈ R^m | b=Ax    for some x ∈ R^n} = CS(A)``
- ``Range(A^T) = {y ∈ R^n | y=A^T*x for some x ∈ R^m} = RS(A)``


Theorem 5.2.1 ``N(A) = Range(A^T)_perp`` and ``N(A^T) = Range(A)_perp``

(see proof on page 235)


Example 3

.. code-block:: text

  A = 1 0
      2 0

  CS(A) = a.(1,2)T

  b=Ax => b=x_1.(1,2)T


What about the null space of A^T?

Theorem 5.2.2 - ``dim(S) + dim(S_perp) = n``. Furthermore,``S u S_perp = span{x1, ... x_n} = R^n``


``dim Range(A) = dim Range(A^T) = rank(A) = r``


Questions 5.2
~~~~~~~~~~~~~

Given three vectors, determine if the third is a linear combination of the first two.

- See if ``v3`` is in ``S_perp`` or not.



5.3 - Least Squares Problems (Optional)
#######################################

**TODO:** review what the ``⊕`` symbol means (page 242 **AND** in an earlier section!)

(**Note:** This section should talk about **non-standard bases**.)


Least squares "curve-fitting" is a common problem. Gauss used it accurately predict planetary motion.


Can be generally modeled as an **overdetermined linear system** (more equations than unknowns - usually *inconsistent*).

We can find an **approximate solution** to ``Ax=b`` by minimizing the residual.

Def: the **residual** is ``r(x) = b -Ax``.

The distance between ``b`` and ``Ax`` is: ``|b - Ax| = |r(x)|``.

We want to find an ``x ∈ R^n`` which minimizes this distance. Minimizing ``|r(x)|`` is the same as minimizing ``|r(x)|^2``.

This solution (input) not only exists, but the resulting vector in the column space (output) is unique.

Example 1 - **TODO:** finish this section



5.4 - Inner Product Spaces
##########################

Def: Inner product on vector space ``V``:

- ``〈x,x〉 > 0, it equals zero if x=0``
- ``〈x,y〉=〈y,x〉for all x,y in V``
- ``〈a*x + b*y, z〉= a*〈x,z〉+ b*〈y,z〉 for all x,y,z in V and scalars a,b``

A vector space with an inner product is called an inner product space.


``R^n`` is a vector space (real vectors)

- Ex: ``〈x,x〉= x^T*y`` (scalar product)
- Ex: ``〈x,y〉= sum(x_i * y_i * w_i)`` (weights, weighted product)


``C[a,b]`` is a vector space (functions)

Ex: ``C[a,b]`` (continuous functions) ``〈f,g〉= Integral(f(x)*g(x) dx, a, b)``, (this is positive for ``〈f,f〉``, since ``f(x)^2 >= 0``)

Ex:``P_n`` (polynomials) ``〈p,q〉= Sum[p(x_i)*q(x_i), i=0 to i=n]``
(see proof it's an inner product on page 255-256. **Note:** can also have a weight term, ``w(x_i)``)

The length, or norm of v is given by ``||v|| = Sqrt[〈v,v〉]``.


Pythagorean Law: if ``u,v`` are orthogonal vectors in an **inner product space** V, then
``||u+v||^2 = ||u||^2 + ||v||^2``.

Proof:

.. code-block:: text

  ||u+v||^2 = 〈u+v, u+v〉
            = 〈u,u〉+ 2〈u,v〉 + 〈v,v〉
            = ||u||^2 + ||v||^2.

Geometrically visualized, this is just a right triangle.


Ex: Over ``C[-1,1]``, ``1`` and ``x`` are orthogonal. What are the respective lengths?

Ex: Over ``C[−π, π]``, define ``〈f,g〉= 1/π  * Integral[f (x)g(x) dx, {x, −π, π}]``.
Then ``|| cos x + sin x || = Sqrt[2]``.

This plays an important role in **Fourier analysis** applications involving trig approximations of functions.


The vector space ``R^(m x n)`` has the **Frobenius Norm**, ``||A\|_F = Sqrt[ Sum[a_ij^2] ]``

Ex 4: Define an inner product over ``P_n`` using inner product defined in previous examples.


Scalar Projections
~~~~~~~~~~~~~~~~~~

Let ``u,v ∈ V`` for inner product space V and ``α ∈ R``, then the **scalar projection** of ``u`` onto ``v`` is: ``α = 〈u,v〉/ ||v||``.

The **vector projection** of ``u`` onto ``v`` is: ``p = α*(v / ||v||) = (〈u,v〉/〈v,v〉)* v``.

Observe: if ``v`` is nonzero and ``p = proj(u onto v)``, then

- ``u - p`` and ``p`` are orthogonal
- ``u = p`` iff ``u`` is a scalar multiple of ``v``

(see proof on page 258)


We use these two observations to prove the **Cauchy-Schwarz inequality**:
``|〈u,v〉| ≤ ||u|| * ||v||``. (Equality holds iff u,v are linearly *dependent*.)

(see proof on page 259)


Norms
~~~~~

A vector space ``V``is said to be a **normed linear space** if for all ``v ∈ V`` there is a real ``||v||`` called the **norm** of ``v``, such that

i. ``||v|| ≥ 0``, with equality iff ``v=0``.
ii. ``||αv||= |α|*||v|| for any scalar α.``
iii. ``||v+w|| ≤ ||v|| + ||w||`` for all ``v, w ∈ V``. (aka, **triangle inequality**)


Theorem 5.4.3 If ``V`` is an IPS, then ``||v|| = Sqrt[〈v,v〉]`` for all ``v ∈ V`` defines a norm on ``V``.
(see proof on page 260... and check your work to HW problem 25 from section 5.4)


Norm-1: ``||x||_1 = Sum[ |x_i|, i=1 to i=n]`` (sum of absolute values of each ``n`` components)

Norm-inf: ``||x||_∞ = max(|x_i|) over i ∈ I`` (max absolute value)

Norm-2: ``||x||_2 = Sqrt[ Sum[ |x_i|^2 ] ]`` (Euclidean norm)

(see page 261 for interesting discussion on orthogonality in ``R^n``)


**Distance** between ``x`` and ``y`` is the real number ``||y - x||``.

Some applications involve finding the "closest" vector in a subspace ``S`` to a given vector ``v`` in a vector space ``V``.
If the norm used for ``V`` is derived from an inner product, then the closet vector can be computed as a vector projection of ``v`` onto the subspace ``S``.
This type of approximation problem is discussed in **section 5.5**.



5.5 - Orthonormal sets
######################

In ``R^2``, we typically use the standard basis ``{e1, e2}`` rather than, say, ``{(2,1)T, (3,5)T}``.

Elements of the standard basis are orthogonal unit vectors.

In the inner product space ``V``, it generally helps to have a set of mutually orthogonal unit vectors.

This is useful not only for finding coordinates of vectors but also in solving least squares problems.

Def: Let ``v1, v2, ..., v_n`` be nonzero vectors in an IPS ``V``. If ``∀i,j ∈ {1, 2, ..., n}: i != j ⇒ 〈v_i, v_j〉 = 0``, then ``{v1, v2, ..., v_n}`` is an **orthogonal set** of vectors.


Ex 1: ``{(1,1,1)T, (2,1,-3)T, (4,-5,1T)}`` is an orthogonal set in ``R^3``, since all 3 dot products are zero.


Theorem 5.5.1: If ``{v1, v2, ..., v_n}`` is an orthogonal set of nonzero vectors in an inner product space V,
then ``v1, v2, ..., v_n`` are linearly independent.

(see proof on page 264)


Def: An **orthonormal** set of vectors is an orthogonal set of unit vectors.

The set ``{u1, u2, ..., u_n}`` is orthonormal iff ``〈u_i, u_j〉 = {1 (if i = j), 0 (if i != j)}``.

Given any orthogonal set of ``n`` nonzero vectors, you can form an orthonormal st by defining
``u_i = (1 / ||v_i||) v_i``. **Note:** How can you verify this is an orthonormal set?

Ex 2: Form an orthonormal set from the vectors in Example 1.


Ex 3: In ``C[−π, π]``, the set ``{1, cos(x), cos(2x), ... cos(n*x)}`` form an orthogonal set:

- ``〈1, cos(kx)〉= 0``
- ``〈cos(jx), cos(kx)〉= 0, if j != k``

Functions ``{cos(x), cos(2x), ..., cos(n*x)}`` are already unit vectors, since
``〈cos(kx), cos(kx)〉= 1/pi * Integral[cos(kx)^2 dx, x=-pi to x=pi] = 1, for all n ∈ I)``

We need only find the unit vector for the function ``1``,
``||1||^2 = 〈1,1〉= 1/pi * Integral[dx, x=-pi to x=pi] = 2``.

Therefore ``1/Sqrt[2]`` is a unit vector, and the set ``{1/Sqrt[2], cos(x), cos(2x), ..., cos(n*x)}`` form an orthonormal set of vectors.


Theorem 5.5.2: Let ``{u1, u2, ..., u_n}`` be an orthonormal basis for an inner produce space ``V``. If ``v = Sum[c_i * u_i, i=1 to i=n]``, then ``c_i = 〈v,u_i〉``.
(see proof on page 265)

Corollary 5.5.3: Let ``{u1, u2, ..., u_n}`` be an orthonormal basis for an inner product space ``V``. If from i=1 to i=n ``u = Sum[a_i * u_i]`` and ``v = Sum[b_i * u_i]``, then ``〈u,v〉= Sum[a_i * b_i]``.
(see proof on page 266)

Corollary 5.5.4 *Parseval's Formula*
if ``{u1, ...., u_n}`` is an orthonormal basis for an IPS ``V`` and ``v = Sum[c_i * u_i, i=1 to n]``, then
``||v||^2 = Sum[c_i^2, i=1 to n]``.

Proof: see corollary 5.5.3


Ex 4: ``u_1 = (1/sqrt[2], 1/sqrt[2])T, u_2 = (1/sqrt[2], -1/sqrt[2])T`` form an orthonormal basis for ``R^2``.If ``x ∈ R^2``, then ``x^T * u_i = (x1 + x2)/sqrt[2], x^T * u_2 = (x1 - x2)/sqrt[2]``.
By theorem 5.5.2, ``x = (x1 + x2)/sqrt[2] * u_1 + (x1 - x2)/sqrt[2] * u_2``.
By corollary 5.5.4, ``||x||^2 = (x1 + x2)^2 / 2 + (x1 - x2)^2  2 = x1^2 + x2^2``.


Ex 5: Given that ``{1/sqrt[2], cos(2x)}`` is an orthonormal set in ``C[−π, π]`` (IPS as in Example 3),
determine the value of ``Integral[sin(x)^4 dx, x, -π, π]`` without using antiderivatives.

Since ``sin(x)^2 = (1 - cos 2x) / 2 = 1/sqrt(2) * 1/sqrt(2) + (-1/2) cos(2x)``,
Parseval's formula gives ``Integral[sin(x)^4 dx, x, -π, π] = π * ||sin(x)^2||^2 = π(1/2 + 1/4) = 3π/4``.

**TODO:** Revisit Example 5.


Orthogonal Matrices
~~~~~~~~~~~~~~~~~~~

Consider ``n x n`` matrices whose column vectors form an orthonormal set in ``R^n``.

Def: An ``n x n`` matrix ``Q`` is an **orthogonal matrix** if its column vectors form an *orthonormal* set in ``R^n``.


Theorem 5.5.5 An ``n x n`` matrix ``Q`` is orthogonal iff ``Q^T * Q = 1``.

Proof: Q is orthogonal iff its column vectors satisfy ``q_i^T * q_j = delta_ij``. Now ``q_i^T * q_j`` is the (i,j) entry of the matrix ``Q^T * Q``. Thus Q is orthogonal iff ``Q^T * Q = 1``.

**TODO:** allude to ``delta_ij``, but it wasn't defined explicitly in the previous section's notes. Revisit theorem 5.5.5 and its proof.


Ex 6: For a fix angle ``a``, the matrix

.. code-block:: text

  Q = cos(a)  -sin(a)
      sin(a)   cos(a)

is orthogonal and

.. code-block:: text

  Q^-1 = Q^T =  cos(a)  sin(a)
               -sin(a)  cos(a)

Properties of Orthogonal Matrices

a. the column vectors of ``Q`` form an orthonormal basis for ``R^n``.
b. ``Q^T * Q = 1``
c. ``Q^T = Q^-1``
d. ``〈Qx,Qy〉=〈x,y〉``
e. ``||Qx||_2 = ||x||_2``


Permutation Matrices
~~~~~~~~~~~~~~~~~~~~

A *permutation matrix* is one formed by reordering the columns of the identity matrix.

Clearly, permutation matrices are orthogonal matrices.

If ``P`` is the permutation matrix obtained by reordering the columns of ``I`` in the order ``(k1, ...., k_n)``, then ``P = (e_k1, ...., e_kn)``.

If ``A`` is an ``m x n`` matrix, then ``AP = (A*e_k1, ...., A*E_kn) = (a_k1, ...., a_kn)``.

Post-multiplication of A by P reorders the columns of A in the order ``(k1, ...., k_n)``, i.e., if

.. code-block:: text

  A = 1 2 3         and P = 0 1 0
      1 2 3                 0 0 1
                            1 0 0

  then

  AP = 3 1 2
       3 1 2

Since ``P = (e_k1, ...., e_kn)`` is orthogonal, it follows that

.. code-block:: text

  P^-1 = P^T = e_k1^T
                 .
                 .
                 .
               e_kn^T

The ``k1`` column of P^T will be ``e1``, the ``k2`` will be ``e2``, and so on.
Thus, ``P^T`` is a permutation matrix. The matrix ``P^T`` can be formed from ``I`` by reordering its rows in the order ``(k1, k2, ...., k_n)``.

Generally, a permutation matrix can be formed from ``I`` by reordering either its rows or its columns.

If ``Q`` is the permutation matrix formed by reordering the rows of ``I`` in the order ``(k1, k2, ..., k_n)`` and ``B`` is an ``n x r`` matrix, then

.. code-block:: text

       e_k1^T           e_k1^T * B       b_k1
         .                 .              .
  QB =   .       * B =     .          =   .
         .                 .              .
       e_kn^T           e_kn^T * B       b_kn

Thus, ``QB`` is a matrix formed by reordering the rows of ``B`` in the order ``(k1, k2, ..., k_n)``.

.. code-block:: text

      0 0 1                     1 1
  Q = 1 0 0       and       B = 2 2
      0 1 0                     3 3

  then

       3 3
  QB = 1 1
       2 2

In general, if ``P`` is an ``n x n`` permutation matrix, pre-multiplication of an ``n x r`` matrix ``B`` by ``P`` reorders the rows of B and *Post-multiplication* of an ``m x n`` matrix ``A`` by ``P`` reorders the column of ``A``.


Orthonormal Sets and Least Squares
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Approximation of Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~


Approximation by Trigonometric Polynomials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


APPLICATION I: Signal Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Discrete Fourier Transform
******************************


The Fast Fourier Transform
**************************



5.6 - Gram-Schmidt orthogonalization
####################################




~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Chapter 6
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

6.1 - Eigenvalues & eigenvectors
################################

Questions for Professor

- Does a double root have significance, i.e., if ``(λ - 2)^2 = 0``?

Notes 6.1
~~~~~~~~~

Ex 1: 30% of married women divorce and 20% of single women marry each year. Find steady state vector.

**TODO:** Review this to understand the inspiration/motivation for eigenvalues.

Def: ``A`` is ``n x n``, then ``λ`` is an **eigenvalue** if there exists a *nonzero* vector ``x`` such that ``Ax=λx``. Note that ``x`` is called the **eigenvector**.

Ex 2: Since ``Ax=3x``, ``λ=3``.

The set of solutions to ``(A-λI)x = 0`` is ``N(A-λI)``
(a subspace of ``R^n``).

So, if ``λ``is an eigenvalue of ``A``, then ``N(A-λI) != 0``.

The eigenspace is ``N(A-λI)``

``(A-λI)x = 0`` has a nontrivial solution iff ``A-λI`` is singular, i.e., if ``det(A-λI) = 0``.

The nth-degree "characteristic" polynomial is ``p(λ) = det(A-λI)``.
The roots are the eigenvalue(s).
Each root has a ``multiplicity``, usually 1.
Including **repeated** and **complex** roots, each characteristic polynomial has ``n`` roots.


Ex 5: In case of **complex** eigenvalues, there are multiple eigenvectors that are equivalent!

**TODO:** revisit application 1 and 2.


Complex Eigenvalues
~~~~~~~~~~~~~~~~~~~

If ``λ=a+bi`` is a solution, so is ``λ=a-bi`` (the complex conjugate).

A matrix can also be complex.

But for real matrices, complex eigenvalues occur in **conjugate pairs**. So do complex **eigenvectors**.

**Note:** If matrices ``A, B`` have complex entries and ``AB`` is defined, then ``Conj(AB) = Conj(A)*Conj(B)`` (see: Exercise 20).

See Example 5.


The Product and Sum of the Eigenvalues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``p(λ) = det(A-λI) = (-1)^n*(λ - λ1)(λ - λ2)...(λ - λ_n)``

``...              = (λ1 - λ)(λ2 - λ)...(λ_n - λ)``.

Therefore,

``λ1 * λ2 * ... * λ_n = p(0) = det(A)``.

We also have the sum as the ``trace`` of ``A``,

``Sum[λ_i] = Sum[a_ii] = tr(A)``.

Ex 6: Observe the properties of ``λ1*λ2`` and ``λ1 + λ2``.


Similar Matrices
~~~~~~~~~~~~~~~~

**Recall:** B is *similar* to A if there is a non-singular matrix ``S`` such that ``B = S^-1 A S``.


Theorem 6.1.1: Let ``A, B`` be ``n x n``. If ``B`` **is similar** to ``A``, then they have the same characteristic polynomial and the **same eigenvalues**.

Proof:

.. code-block:: text

  p_B(λ) = det(B - λI)
         = det(S^-1 A S - λI)
         = det(S^-1(A - λI)S)
         = det(S^-1)det(A - λI)det(S)
         = det(A - λI)
         = p_A(λ)

Since they have the same characteristic equation, they have the same roots (same eigenvalues).


Ex 7: Check that ``S`` and ``T`` matrices have the same eigenvalues.



6.3 - Diagonalization
#####################

We want to factor ``A=XDX^-1``, where ``D`` is a diagonal matrix.

Theorem 6.3.1: For ``k`` **distinct** eigenvalues of an ``n x n`` matrix, the ``k`` **eigenvectors are linearly independent**.
(see proof on page 328)


Theorem 6.3.2: An ``n x n`` matrix is diagonalizable iff it has ``n`` linearly independent eigenvectors.
(see proof on page 328)


**Remarks**

1. If ``A`` is diagonalizable, then the columns of ``X`` are eigenvectors of ``A`` and the diagonal elements of ``D`` are the corresponding eigenvalues of ``A``.
2. The matrix ``X`` is **not unique**. Reordering columns of ``X`` produces a new ``D``.
3. If ``A`` is ``n x n`` and has ``n`` *distinct* eigenvalues, then ``A`` is diagonalizable.

   a) If they are **not distinct**, then the diagonalizability of ``A`` depends on the linear independence of the eigenvectors.

4. If ``A`` is diagonalizable, then ``A`` can be factored to ``XDX^-1``.


In general, ``A^k = X D^k X^-1``.

It's easy to compute powers of ``A`` based on ``D``.


Ex 2: If every eigenvalue ``λ_i ∈ {0, 1}`` is either zero or one, then ``A^k=A``.


If it has fewer than ``n`` linearly independent eigenvectors, we call the matrix **defective**.
From Theorem 6.3.2, we know a defective matrix is **not** diagonalizable.

Ex 3: If ``A = ((1, 1), (0, 1))``, then both eigenvalue are ``1``. Since any eigenvector corresponding to ``λ = 1`` are a multiple of ``x1 = (1,0)ᵀ``, A is defective and cannot be diagonalized.


Ex 4:

- ``A`` is defective since it has only two linearly independent eigenvectors
- ``B`` is not, since its double root has an eigenspace of dimension 2.

Having the algebraic multiplicity not exceed the geometric multiplicity is key in maintaining diagonalizability.


APPLICATION 1: Markov Chains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**TODO:** Finish remaining (optional) content for this section.



6.5 - The Single-Value Decomposition
####################################
