import os, random, sys
import runCmdAsJob #NOTE: happy to share this code with you if you want but it does just what it sounds like

baseDir = "/pine/scr/d/s/dschride/data/popGenCnn/introgression/drosophila/msmodifiedSimsLessMig/" #NOTE: change to wherever you want to save stuff
os.system("mkdir -p %s" %(baseDir))

"""
AIC: 10303.1237186
with u = 3.500000e-09
Nref: 487835.088398
nu1_0 : 117605.344694
nu2_0 : 4878347.23386
nu1 : 9279970.1758
nu2 : 26691.779717
T : 86404.6219829
2Nref_m12 : 0.0128753943002
2Nref_m21 : 0.0861669095413
"""

sampleSize1=20
sampleSize2=14
numSites=10000
thetaMean, thetaOverRhoMean, nu1Mean, nu2Mean, m12Times2Mean, m21Times2Mean, TMean = 68.29691232, 0.2, 19.022761, 0.054715, 0.025751, 0.172334, 0.664194

sampleNumberPerBatch = 10000
batchNumber = 100

def drawUnif(m, fold=0.5):
    x = m*fold
    return random.uniform(m-x, m+x)

def drawParams(thetaMean, thetaOverRhoMean, nu1Mean, nu2Mean, m12Times2Mean, m21Times2Mean, TMean, mig=False):
    theta = drawUnif(thetaMean)
    thetaOverRho = drawUnif(thetaOverRhoMean)
    rho = theta/thetaOverRho
    nu1 = drawUnif(nu1Mean)
    nu2 = drawUnif(nu2Mean)
    T = drawUnif(TMean)
    m12 = drawUnif(m12Times2Mean)
    m21 = drawUnif(m21Times2Mean)
    if mig:
        #migTime = random.uniform(0, T/4)
        migTime = random.uniform(0, T/16) #going less oldly
        #migProb = random.uniform(0.1, 0.5)
        migProb = random.uniform(0.01, 0.25) #going less bigly
        return theta, rho, nu1, nu2, T, m12, m21, migTime, migProb
    else:
        return theta, rho, nu1, nu2, T, m12, m21

def writeTbsFile(params, outFileName):
    with open(outFileName, "w") as outFile:
        for paramVec in params:
            outFile.write(" ".join([str(x) for x in paramVec]) + "\n")

maxRandMs = 2**32-1
maxRandMsMod = 2**16-1
for batch in range(batchNumber):
    noMigParams, mig12Params, mig21Params = [], [], []
    for i in range(sampleNumberPerBatch):
        seed = random.randint(0, maxRandMs)
        theta, rho, nu1, nu2, splitTime, m12, m21 = drawParams(thetaMean, thetaOverRhoMean, nu1Mean, nu2Mean, m12Times2Mean, m21Times2Mean, TMean)
        #paramVec = [theta, rho, nu1, nu2, m12, m21, splitTime, splitTime]
        paramVec = [theta, rho, nu1, nu2, 0, 0, splitTime, splitTime, seed]
        noMigParams.append(paramVec)

        seed1, seed2, seed3 = [random.randint(0, maxRandMsMod) for i in range(3)]
        theta, rho, nu1, nu2, splitTime, m12, m21, migTime, migProb = drawParams(thetaMean, thetaOverRhoMean, nu1Mean, nu2Mean, m12Times2Mean, m21Times2Mean, TMean, mig=True)
        paramVec = [theta, rho, nu1, nu2, 0, 0, splitTime, splitTime, migTime, 1-migProb, migTime, seed1, seed2, seed3]
        mig12Params.append(paramVec)

        seed1, seed2, seed3 = [random.randint(0, maxRandMsMod) for i in range(3)]
        theta, rho, nu1, nu2, splitTime, m12, m21, migTime, migProb = drawParams(thetaMean, thetaOverRhoMean, nu1Mean, nu2Mean, m12Times2Mean, m21Times2Mean, TMean, mig=True)
        paramVec = [theta, rho, nu1, nu2, 0, 0, splitTime, splitTime, migTime, 1-migProb, migTime, seed1, seed2, seed3]
        mig21Params.append(paramVec)

    noMigOutDir = baseDir + "/noMig_%s" %(batch)
    os.system("mkdir -p %s" %(noMigOutDir))
    noMigTbsFileName = "noMig.tbs"
    #NOTE: the command below uses ms (http://home.uchicago.edu/~rhudson1/source/mksamples.html)
    noMigSimCmd = "cd %s; ms %d %d -t tbs -r tbs %d -I 2 %d %d -n 1 tbs -n 2 tbs -eg 0 1 6.576808 -eg 0 2 -7.841388 -ma x tbs tbs x -ej tbs 2 1 -en tbs 1 1 -seed tbs < %s" %(noMigOutDir, sampleSize1+sampleSize2, sampleNumberPerBatch, numSites, sampleSize1, sampleSize2, noMigTbsFileName)
    writeTbsFile(noMigParams, noMigOutDir + "/" + noMigTbsFileName)
    cmd = "%s > %s/noMig.msOut" %(noMigSimCmd, noMigOutDir)
    runCmdAsJob.runCmdAsJobWithoutWaitingWithLog(cmd, "noMig", "noMig.txt", "600:00", "general", "1G", "%s/noMig.log" %(noMigOutDir))

    mig12OutDir = baseDir + "/mig12_%s" %(batch)
    os.system("mkdir -p %s" %(mig12OutDir))
    mig12TbsFileName = "mig12.tbs"
    #TODO: this uses the archie folks' msmodified program which is on their github i think, so download and compile that and adjust the path below
    mig12SimCmd = "cd %s; /nas/longleaf/home/dschride/githubStuff/ArchIE/msmodified/ms %d %d -t tbs -r tbs %d -I 2 %d %d -n 1 tbs -n 2 tbs -eg 0 1 6.576808 -eg 0 2 -7.841388 -ma x tbs tbs x -ej tbs 2 1 -en tbs 1 1 -es tbs 1 tbs -ej tbs 3 2 -seeds tbs tbs tbs < %s" %(mig12OutDir, sampleSize1+sampleSize2, sampleNumberPerBatch, numSites, sampleSize1, sampleSize2, mig12TbsFileName)
    writeTbsFile(mig12Params, mig12OutDir + "/" + mig12TbsFileName)
    cmd = "%s > %s/mig12.msOut" %(mig12SimCmd, mig12OutDir)
    runCmdAsJob.runCmdAsJobWithoutWaitingWithLog(cmd, "mig12", "mig12.txt", "600:00", "general", "1G", "%s/mig12.log" %(mig12OutDir))

    mig21OutDir = baseDir + "/mig21_%s" %(batch)
    os.system("mkdir -p %s" %(mig21OutDir))
    mig21TbsFileName = "mig21.tbs"
    #TODO: again, the path to msmodified below:
    mig21SimCmd = "cd %s; /nas/longleaf/home/dschride/githubStuff/ArchIE/msmodified/ms %d %d -t tbs -r tbs %d -I 2 %d %d -n 1 tbs -n 2 tbs -eg 0 1 6.576808 -eg 0 2 -7.841388 -ma x tbs tbs x -ej tbs 2 1 -en tbs 1 1 -es tbs 2 tbs -ej tbs 3 1 -seeds tbs tbs tbs < %s" %(mig21OutDir, sampleSize1+sampleSize2, sampleNumberPerBatch, numSites, sampleSize1, sampleSize2, mig21TbsFileName)
    writeTbsFile(mig21Params, mig21OutDir + "/" + mig21TbsFileName)
    cmd = "%s > %s/mig21.msOut" %(mig21SimCmd, mig21OutDir)
    runCmdAsJob.runCmdAsJobWithoutWaitingWithLog(cmd, "mig21", "mig21.txt", "600:00", "general", "1G", "%s/mig21.log" %(mig21OutDir))
