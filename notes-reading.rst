***************************
 Linear Algebra, edition 9
***************************

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sample Exam II (questions)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- 4(b) negative signs on span/basis (first element)
- 8(a)(b) [see Section 3.6 questions]




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

v1=(1,2,3), v2=(2,-1,0), v3=(-2,3,-1), v4=(-1,2,3)

Attempt 1

v1 - v4 = (2,0,0)

v2 - (v1 - v4) = (0,-1,0)

v3 + (v1 - v4) - 3[v2 - (v1 - v4)] = (0,0,-1)


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

Thm 3.4.1 - any collection of ``n+1`` vectors (in a space ``V`` spanned by ``n`` vectors) are L.D.
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

Thm 3.6.1 - Two row equivalent matrices have the same row space.

Def: **rank** is the dimension of the row space

Thm 3.6.2 - ``Ax = b`` is consistent <=> ``b ∈ C(A)`` (b in col space of A)

- ``Ax = 0`` has trivial solution ``x=0`` iff col vectors of A are L.I.

Thm 3.6.2 - ``Ax = b`` is consistent for every ``b ∈ R^m`` iff col vectors span ``R^m``

- ``Ax = b`` has at most one solution for every ``b ∈ R^m`` iff the col vectors of A are L.I.

**NOTE:** if col vectors span ``R^m``, then ``n>=m`` (at least as many rows as columns).

Corollary 3.6.4 - ``n x n`` square matrix ``A`` is nonsingular iff col vectors of ``A`` form a basis for ``R^n``.

Thm 3.6.5 - Let ``A`` be an  ``(m x n)`` matrix, then ``rank(A) + nullity(A) = n``

Thm 3.6.6 - ``dim(R(A)) = dim(C(A))`` (see proof on page 176)


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

Thm 4.1.1 - ``Ker(L)`` is a subspace of ``V``, and ``Range(S)`` is a subspace of ``W``


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

Thm 4.2.1 ``L: R^n -> R^m``, there exists a matrix ``L(x) = Ax`` where ``A`` is an ``m x n`` matrix.

(proof: see page 195)

**Review:** Examples 4, 5, and 6




~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Chapter 5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


5.1 - Scalar Product
####################

Def: ``xT*y = x1*y1 + ... + x_n*y_n``

Distance from x to y: ``|x - y|``

Thm 5.1.1 ``xT*y = |x| |y| cos(theta)`` (for ``R^2`` and ``R^3``)

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
- ``Range(A^T) = {y ∈ R^n | y=A^T*x for some x ∈ R^m} = RS(A)


Thm 5.2.1 ``N(A) = Range(A^T)_perp`` and ``N(A^T) = Range(A)_perp

(see proof on page 235)


Example 3

```text

A = 1 0
    2 0

CS(A) = a.(1,2)T

b=Ax => b=x1.(1,2)T
```

What about the null space of A^T?

Thm 5.2.2 - ``dim(S) + dim(S_perp) = n``. Furthermore,``S u S_perp = span{x1, ... x_n} = R^n``


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

- Ex: ``〈f,g〉= Integral(f(x) g(x) dx, a, b)``, (this is positive for ``〈f,f〉``, since ``f(x)^2 > 0``



5.5 - Orthonormal sets
######################

Hi



5.6 - Gram-Schmidt orthogonalization
####################################

Hi




~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Chapter 6
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

6.1 - Eigenvalues & eigenvectors
################################

Hi



6.2 - Diagonalization
#####################

Hi



6.5 - The Single-Value Decomposition
####################################

Hi
