"""M3C 2018 Homework 1."""
"""Manlin Chawla 01205586"""

# Import modules needed throughout code
import numpy as np
import matplotlib.pyplot as plt

def simulate1(N, Nt, b, e):
    # Setting bounds and error messages for inputs
    assert N > 5 and N % 2 == 1, "N must be an odd positive integer greater than 5"
    assert b > 1, "Parameter b must be greater than 1"
    assert 0 < e < 1, "Parameter e must be in the range 0<e<<1"

    # Initial configuration of S, M=0, C=1
    S = np.ones((N, N), dtype=int)
    j = int((N-1)/2)
    S[j-1:j+2, j-1:j+2] = 0

    # Preallocate matrix for:
    # Fitness scores, C probabilities and fractions of points which are C
    fitnessscore = np.zeros((N, N))
    cprob = np.zeros((N, N))
    fc = np.zeros(Nt+1)
    fc[0] = S.sum()/(N*N)

    # Generate a matrix contaning random probabilities
    # Use later to decide which village changes affiliation
    randomprob = np.random.rand(Nt+1, N, N)

    # Iterate over years Nt
    for year in range(Nt+1):

        # Iterate over coordinates in matrix Sto calcuate fitness scores
        for i in range(N):
            for j in range(N):
                neighbours = S[max(0, i-1):min(i+1, N)+1, max(j-1, 0):min(N, j+1)+1]
                numofneighbours = neighbours.size-1
                sumofneighbours = neighbours.sum()-S[i, j]

                # Points for C villages
                if S[i, j] == 1:
                    cpoints = sumofneighbours
                    fitnessscore[i, j] = cpoints/numofneighbours
                # Points for M villages
                else:
                    mpoints = (sumofneighbours*b)+(numofneighbours-sumofneighbours)*e
                    fitnessscore[i, j] = mpoints/numofneighbours

        # Iterate over coordinates in matrix S to calculate C probabilities
        for i in range(N):
            for j in range(N):
                neighbours = S[max(0, i-1):min(i+1, N)+1, max(j-1, 0):min(N, j+1)+1]
                fitnessneighbours = fitnessscore[max(0, i-1):min(i+1, N)+1, max(j-1, 0):min(N, j+1)+1]

                # Totalfitness of all villaged in the community
                totalfitness = fitnessneighbours.sum()
                # Total fitness of C villages in community
                cfitness = fitnessneighbours[neighbours == 1].sum()
                # Probability of being a C the following year
                cprob[i, j] = cfitness/totalfitness

        # New configuration of S based on the probabilities
        S = (randomprob[year, :, :] < cprob).astype(np.int)
        # Calculating fraction of villages which are now C
        fc[year] = S.sum()/(N*N)

        # Break if the fractions are 1 or 0 i.e all villages are C or M
        # Update the rest of the fractions
        if (fc[year] == 1) | (fc[year] == 0):
            fc[year+1:Nt+1] = fc[year]
            break

    return S, fc

def plot_S(S):
    """Simple function to create plot from input S matrix"""
    ind_s0 = np.where(S==0) #C locations
    ind_s1 = np.where(S==1) #M locations
    plt.plot(ind_s0[1], ind_s0[0], 'rs')
    plt.hold(True)
    plt.plot(ind_s1[1], ind_s1[0], 'bs')
    plt.hold(False)
    plt.show()
    plt.pause(0.05)
    return None

def simulate2(N, Nt, b, e):
    # Setting bounds and error messages for inputs
    assert N > 5 and N % 2 == 1, "N must be an odd positive integer greater than 5"
    assert b > 1, "Parameter b must be greater than 1"
    assert 0 < e < 1, "Parameter e must be in the range 0<e<<1"

    # Initial configuration of S, M=0, C=1
    S = np.ones((N, N), dtype=int)
    j = int((N-1)/2)
    S[j-1:j+2, j-1:j+2] = 0

    # Preallocate matrix for:
    # Fitness scores, C probabilities and fractions of points which are C
    fitnessscore = np.zeros((N, N))
    cprob = np.zeros((N, N))
    fc = np.zeros(Nt+1)
    fc[0] = S.sum()/(N*N)

    # Generate a matrix contaning random probabilities
    # Use later to decide which village changes affiliation
    randomprob = np.random.rand(Nt+1, N, N)

    # Iterate over years Nt
    for year in range(Nt+1):

        # Iterate over coordinates in matrix Sto calcuate fitness scores
        for i in range(N):
            for j in range(N):
                neighbours = S[max(0, i-1):min(i+1, N)+1, max(j-1, 0):min(N, j+1)+1]
                numofneighbours = neighbours.size-1
                sumofneighbours = neighbours.sum()-S[i, j]

                # Points for C villages
                if S[i, j] == 1:
                    cpoints = sumofneighbours
                    fitnessscore[i, j] = cpoints/numofneighbours
                # Points for M villages
                else:
                    mpoints = (sumofneighbours*b)+(numofneighbours-sumofneighbours)*e
                    fitnessscore[i, j] = mpoints/numofneighbours

        # Iterate over coordinates in matrix S to calculate C probabilities
        for i in range(N):
            for j in range(N):
                neighbours = S[max(0, i-1):min(i+1, N)+1, max(j-1, 0):min(N, j+1)+1]
                fitnessneighbours = fitnessscore[max(0, i-1):min(i+1, N)+1, max(j-1, 0):min(N, j+1)+1]

                # Totalfitness of all villaged in the community
                totalfitness = fitnessneighbours.sum()
                # Total fitness of C villages in community
                cfitness = fitnessneighbours[neighbours == 1].sum()
                # Probability of being a C the following year
                cprob[i, j] = cfitness/totalfitness

        # New configuration of S based on the probabilities
        S = (randomprob[year, :, :] < cprob).astype(np.int)
        # Calculating fraction of villages which are now C
        fc[year] = S.sum()/(N*N)

        # Break if the fractions are 1 or 0 i.e all villages are C or M
        # Update the rest of the fractions
        if(fc[year] == 1) | (fc[year] == 0):
            fc[year+1:Nt+1] = fc[year]
            break

    # Only returns fc and the year
    return fc, year

def analyze(Nt, figurenum):
    assert figurenum <= 4, "Choose from figures 1,2,3 or 4"

    """ The function analyze code is split into four sections each generating a
    figure that is used to explore and illustrate the key qualitative trends
    (when N=21 e=0.01). The functions simulate1/simulate2 use randomly
    generated probabilities that determine whether a village changes affiliation
    or not, this makes the process stochastic. The output of each figure cannot be
    predicted precisely so my analysis is based on multiple runs of each figure
    describing the overall trend appearing.

    Figure 1: The two plots in this figure show the number of iterations (years) it
    takes for villages in N x N grid (matrix S) to converge to either all
    Collaborators or all Mercenaries for 100 values in the range 1.0<b<=1.5 over
    500 iterationss. The figure shows in general, there is a cluster of values in
    the (approx) range 1<b<=1.15 where all of the villages turn to Collaborators
    usually under 100 iterations. Occasionally there are some values within
    this cluster which do not converge to Collaborators or Mercenaries after 500
    iterations. This is represented by the points along the top of plot 1. The
    majority of the values in the (approx) range 1.1<=b<=1.5 have all of the
    villages converging to Mercernaries.

    Figure 2: This figure shows the fraction of villages which are Collaborator's
    over 200 iterations for 6 values of b 1.05<=b<=1.1 (as b is greater than 1).
    The key qualitative trends are in general b=1.05 is the only value of b out
    of the 10 where fc settles to 1 i.e all villages Collaborator's.In contrast,
    for 1.1<=b<=1.5 the fc settles to 0 i.e all the villages are
    Mercenaries.

    Figure 3: This figure shows the fraction of villages which are Collaborator's
    over 200 iterations for 6 values of b 1.01<=b<=1.11 (as b is greater than 1).
    A recurring trend in this figure was that extremely small values of b (closer
    to 1)have very little fluctuation in fc and stabilize relatively quickly to fc=1.
    Roughly all of the villages are Collaborators within the first 50 iterations
    or less. For the larger values of b (closer to 1.11) the fc become relatively
    stable after 100 iterations and onwards. Intermediate values of b usually have
    have a lot more fluctuation and some take longer than 200 iterations to converge.
    It is surprising to see that sometimes the larger values of b (closer to 1.1)
    converge to fc =1. However, this is not unreasonable as figure 1 also shows that
    some values of small b converge to Collaborators even if the surrounding values
    tend to Mercernaries.

    Figure 4: This figure shows the fraction of villages which are Collaborator's
    over 200 iterations for 6 values of b 1.4<=b<=1.5. For these values of b the
    general qualitative trend is that fc tends to 1 meaning all of the villages
    become Mercernaries.

    Main Conclusion: All of the figures generated support the claim that as b
    increases M becomes more and more successful. Figure 1 and 2 show that there
    are some small b for which all villages become C but as b is increased all
    of the fc converge to M (shown in figure 4 as well). The second plot (of
    Figure 1) and Figure 2 makes it clear to see that as b increases M becomes
    more and more successful. However,the number of years taken for convergence
    dramatically decreases for smaller values of b but this rate slows as b reaches
    larger values. Figure 3 supports this as there a large discrepencies in the
    convergence time amongst the values of b. Whereas, in Figure 4 there is not
    as much dicrepancy in convergence time for amongst the larger values of b. As
    the value of b increases it can be seen from the lines tend to have a steeper
    descent and less flucatuation indicating that the fractions change at a faster
    rate from the initial configuration to 0 and also stabilize much quicker. This
    indicates that as b increases M becomes more and more successful at a faster
    rate. However, the figures do not always show a perfect relationship e.g there
    are some cases b=1.3 takes longer to converge than b=1.2. This could be
    explained as an effect of the stochastic behaviour"""

    # Initialize parameters
    N = 21
    e = 0.01

    # Figure 1
    if figurenum == 1:

        # Preallocate empty matrix for plotting
        cvalue = []
        mvalue = []
        cconvergencetime = []
        mconvergencetime = []

        # Generate values of b, fc and convergence times to plot
        valuesofb = np.linspace(1.005, 1.5, 100)
        for b in valuesofb:
            [fc, year] = simulate2(N, Nt, b, e)
            # Sort points representing convergence to Collaborators or Mercernaries
            if fc[year] == 0:
                mvalue.append(b)
                mconvergencetime.append(year)
            else:
                cvalue.append(b)
                cconvergencetime.append(year)

        plt.hold(True)
        # Subplot 1
        plt.subplot(2, 1, 1)
        plt.plot(cvalue, cconvergencetime, 'b.', label='Collaborators')
        axes = plt.gca()
        axes.set_xlim([1.0, 1.5])
        axes.set_ylim([0, Nt])
        plt.title('Manlin Chawla:analyze(500, 1) \n Convergence time for S to converge to Collaborators or Mercenaries')
        plt.xlabel('Parameter (b)')
        plt.ylabel('no. of iterations \n (years)')
        plt.legend()
        # Adjust space between subplots
        plt.subplots_adjust(hspace=0.50)
        # Subplot 2
        plt.subplot(2, 1, 2)
        plt.plot(mvalue, mconvergencetime, 'r.', label='Mercenaries')
        axes = plt.gca()
        axes.set_xlim([1.0, 1.5])
        axes.set_ylim([0, Nt])
        plt.xlabel('Parameter (b)')
        plt.ylabel('no. of iterations \n (years)')
        plt.legend()

        plt.hold(False)
        #plt.show()

    # Figure 2
    if figurenum == 2:
        # Plot of fc against iterations (years) for different values of b
        valuesofb = np.linspace(1.05, 1.5, 10)
        plt.hold(True)
        for b in valuesofb:
            fc = simulate2(N, Nt, b, e)[0]
            plt.plot(fc, label='b='+str(b))
        plt.title('Manlin Chawla:analyze(200,2) \n Fraction of villages which are Collaborators after each iteration (year)')
        plt.xlabel('no. of iterations (years)')
        plt.ylabel('Fraction of villages \n which are Collaborators (fc)')
        plt.legend()
        plt.hold(False)
        #plt.show()

    # Figure 3
    if figurenum == 3:
        # Plot of fc against iterations (years) for different values of b
        valuesofb = np.linspace(1.01, 1.11, 6)
        plt.hold(True)
        for b in valuesofb:
            fc = simulate2(N, Nt, b, e)[0]
            plt.plot(fc, label='b='+str(b))
        plt.title('Manlin Chawla:analyze(200,3) \n Fraction of villages which are Collaborators after each iteration (year)')
        plt.xlabel('no. of iterations (years)')
        plt.ylabel('Fraction of villages \n which are Collaborators (fc)')
        plt.legend()
        plt.hold(False)
        #plt.show()

    # Figure 4
    if figurenum == 4:
        # Plot of fc against iterations(years) for different values of b
        valuesofb = np.linspace(1.4, 1.5, 6)
        plt.hold(True)
        for b in valuesofb:
            fc = simulate2(N, Nt, b, e)[0]
            plt.plot(fc, label='b='+str(b))
        plt.title('Manlin Chawla:analyze(200,4) \n Fraction of villages which are Collaborators after each iteration (year)')
        plt.xlabel('no. of iterations (years)')
        plt.ylabel('Fraction of villages \n which are Collaborators (fc)')
        plt.legend()
        plt.hold(False)
        #plt.show()

if __name__ == '__main__':
    # The code calls analyze and generates figures that are submitted

    # Generates figure 1, formats title and axis labels
    output = analyze(500, 1)
    plt.savefig('hw11.png', bbox_inches="tight")
    plt.show()
    plt.clf()

    # Generates figure 2, formats title and axis labels
    output = analyze(200, 2)
    plt.savefig('hw12.png', bbox_inches="tight")
    plt.show()
    plt.clf()

    # Generates figure 3, formats title and axis labels
    output = analyze(200, 3)
    plt.savefig('hw13.png', bbox_inches="tight")
    plt.show()
    #plt.clf()

    # Generates figure 4, formats title and axis labels
    output = analyze(200, 4)
    plt.savefig('hw14.png', bbox_inches="tight")
    plt.show()
    plt.clf()
