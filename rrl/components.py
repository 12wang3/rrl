import torch
import torch.nn as nn
from collections import defaultdict

THRESHOLD = 0.5
INIT_RANGE = 0.5
EPSILON = 1e-10
INIT_L = 0.0


class GradGraft(torch.autograd.Function):
    """Implement the Gradient Grafting."""
    @staticmethod
    def forward(ctx, X, Y):
        return X

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output.clone()


class Binarize(torch.autograd.Function):
    """Deterministic binarization."""
    @staticmethod
    def forward(ctx, X):
        y = torch.where(X > 0, torch.ones_like(X), torch.zeros_like(X))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BinarizeLayer(nn.Module):
    """Implement the feature discretization and binarization."""

    def __init__(self, n, input_dim, use_not=False, left=None, right=None):
        super(BinarizeLayer, self).__init__()
        self.n = n
        self.input_dim = input_dim
        self.disc_num = input_dim[0]
        self.use_not = use_not
        if self.use_not:
            self.disc_num *= 2
        self.output_dim = self.disc_num + self.n * self.input_dim[1] * 2
        self.layer_type = 'binarization'
        self.dim2id = {i: i for i in range(self.output_dim)}
        self.rule_name = None

        self.register_buffer('left', left)
        self.register_buffer('right', right)

        if self.input_dim[1] > 0:
            if self.left is not None and self.right is not None:
                cl = self.left + torch.rand(self.n, self.input_dim[1]) * (self.right - self.left)
            else:
                cl = torch.randn(self.n, self.input_dim[1])
            self.register_buffer('cl', cl)

    def forward(self, x):
        if self.input_dim[1] > 0:
            x_disc, x = x[:, 0: self.input_dim[0]], x[:, self.input_dim[0]:]
            x = x.unsqueeze(-1)
            if self.use_not:
                x_disc = torch.cat((x_disc, 1 - x_disc), dim=1)
            binarize_res = Binarize.apply(x - self.cl.t()).view(x.shape[0], -1)
            return torch.cat((x_disc, binarize_res, 1. - binarize_res), dim=1)
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        return x

    @torch.no_grad()
    def binarized_forward(self, x):
        return self.forward(x)

    def clip(self):
        if self.input_dim[1] > 0 and self.left is not None and self.right is not None:
            self.cl.data = torch.where(self.cl.data > self.right, self.right, self.cl.data)
            self.cl.data = torch.where(self.cl.data < self.left, self.left, self.cl.data)

    def get_bound_name(self, feature_name, mean=None, std=None):
        bound_name = []
        for i in range(self.input_dim[0]):
            bound_name.append(feature_name[i])
        if self.use_not:
            for i in range(self.input_dim[0]):
                bound_name.append('~' + feature_name[i])
        if self.input_dim[1] > 0:
            for c, op in [(self.cl, '>'), (self.cl, '<=')]:
                c = c.detach().cpu().numpy()
                for i, ci in enumerate(c.T):
                    fi_name = feature_name[self.input_dim[0] + i]
                    for j in ci:
                        if mean is not None and std is not None:
                            j = j * std[fi_name] + mean[fi_name]
                        bound_name.append('{} {} {:.3f}'.format(fi_name, op, j))
        self.rule_name = bound_name


class Product(torch.autograd.Function):
    """Tensor product function."""
    @staticmethod
    def forward(ctx, X):
        y = (-1. / (-1. + torch.sum(torch.log(X), dim=1)))
        ctx.save_for_backward(X, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        X, y, = ctx.saved_tensors
        grad_input = grad_output.unsqueeze(1) * (y.unsqueeze(1) ** 2 / (X + EPSILON))
        return grad_input


class EstimatedProduct(torch.autograd.Function):
    """Tensor product function with a estimated derivative."""
    @staticmethod
    def forward(ctx, X):
        y = (-1. / (-1. + torch.sum(torch.log(X), dim=1)))
        ctx.save_for_backward(X, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        X, y, = ctx.saved_tensors
        grad_input = grad_output.unsqueeze(1) * ((-1. / (-1. + torch.log(y.unsqueeze(1) ** 2))) / (X + EPSILON))
        return grad_input


class LRLayer(nn.Module):
    """The LR layer is used to learn the linear part of the data."""

    def __init__(self, n, input_dim):
        super(LRLayer, self).__init__()
        self.n = n
        self.input_dim = input_dim
        self.output_dim = self.n
        self.layer_type = 'linear'
        self.rid2dim = None
        self.rule2weights = None

        self.fc1 = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        return self.fc1(x)

    @torch.no_grad()
    def binarized_forward(self, x):
        return self.forward(x)

    def clip(self):
        for param in self.fc1.parameters():
            param.data.clamp_(-1.0, 1.0)
        
    def l1_norm(self):
        return torch.norm(self.fc1.weight, p=1)
    
    def l2_norm(self):
        return torch.sum(self.fc1.weight ** 2)
    
    def get_rule2weights(self, prev_layer, skip_connect_layer):
        prev_layer = self.conn.prev_layer
        skip_connect_layer = self.conn.skip_from_layer

        always_act_pos = (prev_layer.node_activation_cnt == prev_layer.forward_tot)
        merged_dim2id = prev_dim2id = {k: (-1, v) for k, v in prev_layer.dim2id.items()}
        if skip_connect_layer is not None:
            shifted_dim2id = {(k + prev_layer.output_dim): (-2, v) for k, v in skip_connect_layer.dim2id.items()}
            merged_dim2id = defaultdict(lambda: -1, {**shifted_dim2id, **prev_dim2id})
            always_act_pos = torch.cat(
                [always_act_pos, (skip_connect_layer.node_activation_cnt == skip_connect_layer.forward_tot)])
        
        Wl, bl = list(self.fc1.parameters())
        bl = torch.sum(Wl.T[always_act_pos], dim=0) + bl
        Wl = Wl.cpu().detach().numpy()
        self.bl = bl.cpu().detach().numpy()

        marked = defaultdict(lambda: defaultdict(float))
        rid2dim = {}
        for label_id, wl in enumerate(Wl):
            for i, w in enumerate(wl):
                rid = merged_dim2id[i]
                if rid == -1 or rid[1] == -1:
                    continue
                marked[rid][label_id] += w
                rid2dim[rid] = i % prev_layer.output_dim

        self.rid2dim = rid2dim
        self.rule2weights = sorted(marked.items(), key=lambda x: max(map(abs, x[1].values())), reverse=True)


class ConjunctionLayer(nn.Module):
    """The novel conjunction layer is used to learn the conjunction of nodes with less time and GPU memory usage."""

    def __init__(self, n, input_dim, use_not=False, alpha=0.999, beta=8, gamma=1):
        super(ConjunctionLayer, self).__init__()
        self.n = n
        self.use_not = use_not
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = self.n
        self.layer_type = 'conjunction'

        self.W = nn.Parameter(INIT_L + (0.5 - INIT_L) * torch.rand(self.input_dim, self.n)) 

        self.node_activation_cnt = None

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, x):
        res_tilde = self.continuous_forward(x)
        res_bar = self.binarized_forward(x)
        return GradGraft.apply(res_bar, res_tilde)

    def continuous_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        x = 1. - x
        xl = (1. - 1. / (1. - (x * self.alpha) ** self.beta))
        wl = (1. - 1. / (1. - (self.W * self.alpha) ** self.beta))
        return 1. / (1. + xl @ wl) ** self.gamma

    @torch.no_grad()
    def binarized_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        Wb = Binarize.apply(self.W - THRESHOLD)
        res = (1 - x) @ Wb
        return torch.where(res > 0, torch.zeros_like(res), torch.ones_like(res))

    def clip(self):
        self.W.data.clamp_(INIT_L, 1.0)


class DisjunctionLayer(nn.Module):
    """The novel disjunction layer is used to learn the disjunction of nodes with less time and GPU memory usage."""

    def __init__(self, n, input_dim, use_not=False, alpha=0.999, beta=8, gamma=1):
        super(DisjunctionLayer, self).__init__()
        self.n = n
        self.use_not = use_not
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = self.n
        self.layer_type = 'disjunction'
        
        self.W = nn.Parameter(INIT_L + (0.5 - INIT_L) * torch.rand(self.input_dim, self.n))

        self.node_activation_cnt = None

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, x):
        res_tilde = self.continuous_forward(x)
        res_bar = self.binarized_forward(x)
        return GradGraft.apply(res_bar, res_tilde)

    def continuous_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        xl = (1. - 1. / (1. - (x * self.alpha) ** self.beta))
        wl = (1. - 1. / (1. - (self.W * self.alpha) ** self.beta))
        return 1. - 1. / (1. + xl @ wl) ** self.gamma

    @torch.no_grad()
    def binarized_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        Wb = Binarize.apply(self.W - THRESHOLD)
        res = x @ Wb
        return torch.where(res > 0, torch.ones_like(res), torch.zeros_like(res))

    def clip(self):
        self.W.data.clamp_(INIT_L, 1.0)


class OriginalConjunctionLayer(nn.Module):
    """The conjunction layer is used to learn the conjunction of nodes."""

    def __init__(self, n, input_dim, use_not=False, estimated_grad=False):
        super(OriginalConjunctionLayer, self).__init__()
        self.n = n
        self.use_not = use_not
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = self.n
        self.layer_type = 'conjunction'

        self.W = nn.Parameter(INIT_RANGE * torch.rand(self.input_dim, self.n))        
        self.Product = EstimatedProduct if estimated_grad else Product
        self.node_activation_cnt = None

    def forward(self, x):
        res_tilde = self.continuous_forward(x)
        res_bar = self.binarized_forward(x)
        return GradGraft.apply(res_bar, res_tilde)
    
    def continuous_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        return self.Product.apply(1 - (1 - x).unsqueeze(-1) * self.W)

    @torch.no_grad()
    def binarized_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        Wb = Binarize.apply(self.W - THRESHOLD)
        return torch.prod(1 - (1 - x).unsqueeze(-1) * Wb, dim=1)

    def clip(self):
        self.W.data.clamp_(0.0, 1.0)


class OriginalDisjunctionLayer(nn.Module):
    """The disjunction layer is used to learn the disjunction of nodes."""

    def __init__(self, n, input_dim, use_not=False, estimated_grad=False):
        super(OriginalDisjunctionLayer, self).__init__()
        self.n = n
        self.use_not = use_not
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = self.n
        self.layer_type = 'disjunction'
        
        self.W = nn.Parameter(INIT_RANGE * torch.rand(self.input_dim, self.n))
        self.Product = EstimatedProduct if estimated_grad else Product
        self.node_activation_cnt = None

    def forward(self, x):
        res_tilde = self.continuous_forward(x)
        res_bar = self.binarized_forward(x)
        return GradGraft.apply(res_bar, res_tilde)

    def continuous_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        return 1 - self.Product.apply(1 - x.unsqueeze(-1) * self.W)

    @torch.no_grad()
    def binarized_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        Wb = Binarize.apply(self.W - THRESHOLD)
        return 1 - torch.prod(1 - x.unsqueeze(-1) * Wb, dim=1)

    def clip(self):
        self.W.data.clamp_(0.0, 1.0)
        
        
def extract_rules(prev_layer, skip_connect_layer, layer, pos_shift=0):
    # dim2id = {dimension: rule_id} : 
    dim2id = defaultdict(lambda: -1)
    rules = {}
    tmp = 0
    rule_list = []

    # Wb.shape = (n, input_dim)
    Wb = (layer.W.t() > 0.5).type(torch.int).detach().cpu().numpy()

    # merged_dim2id is the dim2id of the input (the prev_layer and skip_connect_layer)
    merged_dim2id = prev_dim2id = {k: (-1, v) for k, v in prev_layer.dim2id.items()}
    if skip_connect_layer is not None:
        shifted_dim2id = {(k + prev_layer.output_dim): (-2, v) for k, v in skip_connect_layer.dim2id.items()}
        merged_dim2id = defaultdict(lambda: -1, {**shifted_dim2id, **prev_dim2id})

    for ri, row in enumerate(Wb):
        # delete dead nodes
        if layer.node_activation_cnt[ri + pos_shift] == 0 or layer.node_activation_cnt[ri + pos_shift] == layer.forward_tot:
            dim2id[ri + pos_shift] = -1
            continue
        rule = {}
        # rule[i] = (k, rule_id): 
        #     k == -1: connects to a rule in prev_layer, 
        #     k ==  1: connects to a rule in prev_layer (NOT),
        #     k == -2: connects to a rule in skip_connect_layer, 
        #     k ==  2: connects to a rule in skip_connect_layer (NOT).
        bound = {}
        if prev_layer.layer_type == 'binarization' and prev_layer.input_dim[1] > 0:
            c = torch.cat((prev_layer.cl.t().reshape(-1), prev_layer.cl.t().reshape(-1))).detach().cpu().numpy()
        for i, w in enumerate(row):
            # deal with "use NOT", use_not_mul = -1 if it used NOT in that input dimension
            use_not_mul = 1
            if layer.use_not:
                if i >= layer.input_dim // 2:
                    use_not_mul = -1
                i = i % (layer.input_dim // 2)
            
            if w > 0 and merged_dim2id[i][1] != -1:
                if prev_layer.layer_type == 'binarization' and i >= prev_layer.disc_num:
                    ci = i - prev_layer.disc_num
                    bi = ci // prev_layer.n
                    if bi not in bound:
                        bound[bi] = [i, c[ci]]
                        rule[(-1, i)] = 1  # since dim2id[i] == i in the BinarizeLayer
                    else:  # merge the bounds for one feature
                        if (ci < c.shape[0] // 2 and layer.layer_type == 'conjunction') or \
                           (ci >= c.shape[0] // 2 and layer.layer_type == 'disjunction'):
                            func = max
                        else:
                            func = min
                        bound[bi][1] = func(bound[bi][1], c[ci])
                        if bound[bi][1] == c[ci]:  # replace the last bound
                            del rule[(-1, bound[bi][0])]
                            rule[(-1, i)] = 1
                            bound[bi][0] = i
                else:
                    rid = merged_dim2id[i]
                    rule[(rid[0] * use_not_mul, rid[1])] = 1
        
        # give each unique rule an id, and save this id in dim2id
        rule = tuple(sorted(rule.keys()))
        if rule not in rules:
            rules[rule] = tmp
            rule_list.append(rule)
            dim2id[ri + pos_shift] = tmp
            tmp += 1
        else:
            dim2id[ri + pos_shift] = rules[rule]
    return dim2id, rule_list


class UnionLayer(nn.Module):
    """The union layer is used to learn the rule-based representation."""

    def __init__(self, n, input_dim, use_not=False, use_nlaf=False, estimated_grad=False, alpha=0.999, beta=8, gamma=1):
        super(UnionLayer, self).__init__()
        self.n = n
        self.use_not = use_not
        self.input_dim = input_dim
        self.output_dim = self.n * 2
        self.layer_type = 'union'
        self.forward_tot = None
        self.node_activation_cnt = None
        self.dim2id = None
        self.rule_list = None
        self.rule_name = None

        if use_nlaf: # use novel logical activation functions
            self.con_layer = ConjunctionLayer(self.n, self.input_dim, use_not=use_not, alpha=alpha, beta=beta, gamma=gamma)
            self.dis_layer = DisjunctionLayer(self.n, self.input_dim, use_not=use_not, alpha=alpha, beta=beta, gamma=gamma)
        else: # use original logical activation functions
            self.con_layer = OriginalConjunctionLayer(self.n, self.input_dim, use_not=use_not, estimated_grad=estimated_grad)
            self.dis_layer = OriginalDisjunctionLayer(self.n, self.input_dim, use_not=use_not, estimated_grad=estimated_grad)

    def forward(self, x):
        return torch.cat([self.con_layer(x), self.dis_layer(x)], dim=1)

    def binarized_forward(self, x):
        return torch.cat([self.con_layer.binarized_forward(x),
                          self.dis_layer.binarized_forward(x)], dim=1)
    
    def edge_count(self):
        con_Wb = Binarize.apply(self.con_layer.W - THRESHOLD)
        dis_Wb = Binarize.apply(self.dis_layer.W - THRESHOLD)
        return torch.sum(con_Wb) + torch.sum(dis_Wb)
    
    def l1_norm(self):
        return torch.sum(self.con_layer.W) + torch.sum(self.dis_layer.W)
    
    def l2_norm(self):
        return torch.sum(self.con_layer.W ** 2) + torch.sum(self.dis_layer.W ** 2)

    def clip(self):
        self.con_layer.clip()
        self.dis_layer.clip()

    def get_rules(self, prev_layer, skip_connect_layer):
        self.con_layer.forward_tot = self.dis_layer.forward_tot = self.forward_tot
        self.con_layer.node_activation_cnt = self.dis_layer.node_activation_cnt = self.node_activation_cnt

        # get dim2id and rule lists of the conjunction layer and the disjunction layer
        # dim2id: dimension --> (k, rule id) 
        con_dim2id, con_rule_list = extract_rules(prev_layer, skip_connect_layer, self.con_layer)
        dis_dim2id, dis_rule_list = extract_rules(prev_layer, skip_connect_layer, self.dis_layer, self.con_layer.W.shape[1])

        shift = max(con_dim2id.values()) + 1
        dis_dim2id = {k: (-1 if v == -1 else v + shift) for k, v in dis_dim2id.items()}
        dim2id = defaultdict(lambda: -1, {**con_dim2id, **dis_dim2id})
        
        rule_list = (con_rule_list, dis_rule_list)

        self.dim2id = dim2id
        self.rule_list = rule_list

    def get_rule_description(self, input_rule_name, wrap=False):
        """
        input_rule_name: (skip_connect_rule_name, prev_rule_name)
        """
        self.rule_name = []
        for rl, op in zip(self.rule_list, ('&', '|')):
            for rule in rl:
                name = ''
                for i, ri in enumerate(rule):
                    op_str = ' {} '.format(op) if i != 0 else ''
                    layer_shift = ri[0]
                    not_str = ''
                    if ri[0] > 0: # ri[0] == 1 or ri[0] == 2
                        layer_shift *= -1
                        not_str = '~'
                    var_str = ('({})' if (wrap or not_str == '~') else '{}').format(input_rule_name[2 + layer_shift][ri[1]])
                    name += op_str + not_str + var_str
                self.rule_name.append(name)
