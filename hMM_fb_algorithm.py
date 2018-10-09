%%
def forward_algorithm(initial_probability, emission_probability, transition_matrix, X_train, observation):
   pi = initial_probability
   O = emission_probability
   S = pi.shape[0]
   A = transition_matrix
   N = len(observation)
   
   alpha = np.zeros((N,S))
   observer_index = []
   for j in range(len(observation)):
       for i in range(len(X_train)):
           if np.array_equal(observation[j], X_train[i]):
               observer_index.append(i)
   #base case
   alpha[0, :] = pi*O[:, observer_index[0]]
   for i in range(1, N):
       for s2 in range(S):
           for s1 in range(S):
               alpha[i, s2]+=alpha[i-1, s1]* A[s1, s2] * O[s2, observer_index[i]]
   
   return (alpha, np.sum(alpha[N-1, :]))

%%
def backward_algorithm(initial_probability, emission_probability, transition_matrix, X_train, observation):
   pi = initial_probability
   O = emission_probability
   S = pi.shape[0]
   A = transition_matrix
   N = len(observation)
   
   beta = np.zeros((N,S))
   observer_index = []
   for j in range(len(observation)):
       for i in range(len(X_train)):
           if np.array_equal(observation[j], X_train[i]):
               observer_index.append(i)
   #base case
   beta[N-1, :] = 1
   for i in range(N-2, -1, -1):
       for s1 in range(S):
           for s2 in range(S):
               beta[i, s1]+=beta[i+1, s2]* A[s1, s2] * O[s2, observer_index[i+1]]
   
   return(beta, np.sum(pi*O[:, observer_index[0]]*beta[0, :]))



def fb_algorithm(build_train_data, X_train, initial_probability, emission_probability, transition_matrix, iterations):
   pi = initial_probability
   A = transition_matrix
   O = emission_probability
   pi_1 = np.copy(pi)
   A_1 = np.copy(A)
   O_1 = np.copy(O)
   S = pi.shape[0]
   for q in range(100):
       pi_2 = np.zeros_like(pi)
       A_2 = np.zeros_like(A)
       O_2 = np.zeros_like(O)
       
       for data in build_train_data["input_data"]:
           #compute the forward-backward matrices
           #Expectation step
           alpha, za = forward_algorithm(pi, O, A, X_train, data)
           print(za)
           beta, zb = backward_algorithm(pi, O, A, X_train, data)
           print(zb)
           assert abs(za - zb) <=0 ,"error less than expected"
           observer_index = []
           for j in range(len(data)):
               for i in range(len(X_train)):
                   if np.array_equal(data[j], X_train[i]):
                       observer_index.append(i)
                       
           #Maximization step
           pi_2 += alpha[0, :]*beta[0,:]/za
           for e in range(len(data)):
               O_2[:, observer_index[e]]+=alpha[e,:]*beta[e,:]/za
           for t in range(1, len(data)):
               for s1 in range(S):
                   for s2 in range(S):
                       A_2[s1, s2]+=alpha[t-1, s1]*A[s1, s2]*O[s2, observer_index[t]]*beta[t, s2]/za
           pi_1 = pi_2/np.sum(pi_1)
           for s in range(S):
               A_1[s, :] = A_2[s, :]/np.sum(A_2[s, :])
               O_1[s, :] = O_2[s, :]/np.sum(O_2[s, :])
           print(q)
       return(pi_1, A_1, O_1)
