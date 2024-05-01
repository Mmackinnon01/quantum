def index_to_config(index, dims):
    config = []
    total_dims = [math.prod(dims[i:]) for i in reversed(range(len(dims)))]
    for dim in reversed(total_dims[:-1]):
        config.append(math.floor(index /dim))
        index = index % dim
    config.append(index%dims[-1])
    return config

def config_to_index(config, dims):
    index = 0
    total_dims = [math.prod(dims[i+1:]) for i in range(len(dims)-1)]
    total_dims += [1]
    for i, state in enumerate(config):
        index += (state * total_dims[i])
    return index

def extend_matrix(matrix, dims):
    for dim in dims:
        matrix = np.kron(matrix, np.eye(dim))
    return matrix

def reorder_list(list, config):
    new_list = []
    for i in config:
        new_list.append(list[i])
    return new_list

def rewrite_matrix(matrix, config, dims):
    new_matrix = np.zeros(matrix.shape)
    new_dims = reorder_list(dims, config)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            i_config = index_to_config(i, dims)
            j_config = index_to_config(j, dims)
            new_i_config = reorder_list(i_config, config)
            new_j_config = reorder_list(j_config, config)
            new_i = config_to_index(new_i_config, new_dims)
            new_j = config_to_index(new_j_config, new_dims)
            new_matrix[new_i][new_j] = matrix[i][j]
    return new_matrix

def tranform_matrix(matrix, config, dims):
    num_old = len([val for val in config if val != -1])
    
    for i, state in enumerate(config):
        if state == -1:
            matrix = extend_matrix(matrix, [dims[i]])
            config[i] = i + num_old
        else:
            num_old -= 1
    return rewrite_matrix(matrix, config, reorder_list(dims, config))
    