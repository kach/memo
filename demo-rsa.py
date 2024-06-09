from memoparse import memo, ic

R = [2, 3] # 10 -> hat, 11 -> glasses + hat
U = [2, 3] # 10 -> hat, 01 -> glasses

@memo
def literal_speaker():
    cast: [speaker]
    given: u_ in U
    given: r_ in R

    speaker: chooses(r in R, wpp=1)
    speaker: chooses(u in U, wpp=(0. if (r == 2 and u == 3) else 1.))
    return E[(speaker.u == u_) and (speaker.r == r_)]

@memo
def l1_listener():
    cast: [listener]
    given: u in U
    given: r_ in R

    listener: thinks[
        speaker: chooses(r in R, wpp=1),
        speaker: chooses(u in U, wpp=(0. if (r == 2 and u == 3) else 1.))
    ]
    listener: observes [speaker.u] is self.u
    listener: chooses(r_ in R, wpp=E[speaker.r == r_])
    return E[ listener.r_ == r_ ]
ic(l1_listener())

@memo
def l2_speaker(beta):
    cast: [speaker, listener]
    given: u_ in U
    given: r_ in R

    speaker: thinks[
        listener: thinks[
            speaker: chooses(r in R, wpp=1),
            speaker: chooses(u in U, wpp=(0. if (r == 2 and u == 3) else 1.))
        ]
    ]

    speaker: chooses(r in R, wpp=1)
    speaker: chooses(u in U, wpp=imagine[
        listener: observes [speaker.u] is self.u,
        listener: chooses(r_ in R, wpp=E[speaker.r == r_]),
        exp(beta * E[listener.r_ == r])
    ])
    return E[(speaker.u == u_) and (speaker.r == r_)]
ic(l2_speaker(3.))
