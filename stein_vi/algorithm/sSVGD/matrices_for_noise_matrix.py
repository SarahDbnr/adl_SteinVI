import jax.numpy as jnp
import jax
def compute_stochastic_correction_old(particles, kernel_fn, kernel_params, random_normal_samples):
    n_particles, d = particles.shape
    Nd = n_particles * d
    
    # Step 1: Calculate the kernel Gram matrix k_bar
    k_bar = jnp.zeros((n_particles, n_particles))
    for i in range(n_particles):
        for j in range(n_particles):
            k_bar = k_bar.at[i, j].set(kernel_fn(particles[i], particles[j], **kernel_params))
    
    # Step 2: Construct the block-diagonal matrix D_K
    D_K = jnp.kron(jnp.eye(d), k_bar / n_particles)
    # Step 3: Compute the Cholesky decomposition of D_K
    L_DK = jnp.linalg.cholesky(D_K)
    
    # Step 4: Generate random normal samples from N(0, I_{Nd x Nd})
    #random_normal_samples = jax.random.normal(rng_key, (Nd,))
    
    # Step 5: Create the permutation matrix P
    P = compute_permutation_matrix(n_particles, d)
    #P_test = compute_permutation_matrix_test(n_particles, d)
    #Create test, that checks that the permutation is correct
    # Step 6: Compute the stochastic correction v^STC
    v_stc = jnp.sqrt(2) * jnp.dot(P.T, jnp.dot(L_DK, random_normal_samples))
    return v_stc.reshape(n_particles, d)

def compute_stochastic_correction(particles, kernel_fn, kernel_params, random_normal_samples):
    n_particles, d = particles.shape

    # Step 1: Calculate the kernel Gram matrix k_bar
    def compute_k_bar(particles):
        pairwise_kernels = jax.vmap(
            lambda x: jax.vmap(lambda y: kernel_fn(x, y, **kernel_params))(particles)
        )(particles)
        return pairwise_kernels / n_particles

    k_bar = compute_k_bar(particles)  # Shape (n_particles, n_particles)

    # Step 2: Compute Cholesky decomposition of k_bar
    L_k_bar = jnp.linalg.cholesky(k_bar)  # Shape (n_particles, n_particles)

    # Step 3: Generate random normal samples and reshape
    Nd = n_particles * d
    epsilon = random_normal_samples.reshape(d, n_particles)  # Shape (d, n_particles)

    # Step 4: Compute V = L_k_bar @ epsilon.T, resulting in (n_particles, d)
    V = L_k_bar @ epsilon.T  # Shape (n_particles, d)

    # Step 5: Transpose V to get (d, n_particles)
    V = V.T  # Shape (d, n_particles)

    # Step 6: Reshape V to (n_particles, d)
    v_stc = V.T  # Shape (n_particles, d)

    # Step 7: Multiply by sqrt(2)
    v_stc = jnp.sqrt(2) * v_stc  # Shape (n_particles, d)
    # jax.debug.print("v_stc_new:\n {}", v_stc)
    return v_stc  # Return shape (n_particles, d)



def compute_permutation_matrix_test(N, d):
    Nd = N * d
    P = jnp.zeros((Nd, Nd))
    for i in range(d):
        for j in range(N):
            P = P.at[i * N + j, j * d + i].set(1)
    return P

def compute_permutation_matrix(N, d):
    #should return the same result as the test matrix but i am not sure if this is the case
    Nd = N * d
    # Create the identity matrix of size Nd x Nd
    identity = jnp.eye(Nd)
    
    # Reshape and transpose to achieve the desired permutation
    P = identity.reshape(N, d, N, d).transpose(1, 0, 2, 3).reshape(Nd, Nd)
    
    return P