import torch


def get_xys():
    return torch.load('./xys.pth')


def get_xyzs(xys, zs):
    return torch.concat([xys, zs], dim=1)


def get_alpha_s(delta, phi, omega):
    return torch.arcsin(torch.cos(delta) * torch.cos(phi) * torch.cos(omega) +
                        torch.sin(delta) * torch.sin(phi))


def get_gamma_s(alpha_s, delta, phi):
    return torch.arccos((torch.sin(delta) - torch.sin(alpha_s) * torch.sin(phi)) /
                        (torch.cos(alpha_s) * torch.cos(phi)))


def get_omega(ST):
    return torch.pi * (ST - 12) / 12


def get_delta(D):
    return torch.arcsin(torch.sin(2 * torch.pi * D / 365) * torch.sin(torch.tensor(2 * torch.pi * 23.45 / 360)))


def get_DNI(H, alpha_s):
    a = get_a(H)
    b = get_b(H)
    c = get_c(H)
    return 1.366 * (a + b * torch.exp(-c / torch.sin(alpha_s)))


def get_a(H):
    return 0.4237 - 0.00821 * torch.square(6 - H)


def get_b(H):
    return 0.5055 + 0.00595 * torch.square(6.5 - H)


def get_c(H):
    return 0.2711 + 0.01858 * torch.square(2.5 - H)


def get_E_field(DNI, A, eta):
    return DNI * torch.sum(A * eta)


# etas_sb = {1: 1 - 0.0250,
#            2: 1 - 0.0157,
#            3: 1 - 0.0152,
#            4: 1 - 0.0167,
#            5: 1 - 0.0179,
#            6: 1 - 0.0179,
#            7: 1 - 0.0179,
#            8: 1 - 0.0168,
#            9: 1 - 0.0152,
#            10: 1 - 0.0158,
#            11: 1 - 0.0253,
#            12: 1 - 0.0364}

etas_sb = {1: 1 - 0.0250 / 2,
           2: 1 - 0.0157 / 2,
           3: 1 - 0.0152 / 2,
           4: 1 - 0.0167 / 2,
           5: 1 - 0.0179 / 2,
           6: 1 - 0.0179 / 2,
           7: 1 - 0.0179 / 2,
           8: 1 - 0.0168 / 2,
           9: 1 - 0.0152 / 2,
           10: 1 - 0.0158 / 2,
           11: 1 - 0.0253 / 2,
           12: 1 - 0.0364 / 2}
