from my_tools import Help_Code_Merge,File


geoms=['o']
attrs=['r']
r0Xs=["r01"]
C1C2CWCYs=["lossless-geom-lossy-attrs"]
isIntra=True
reGenerate=True
for geom in geoms:
    for attr in attrs:
        for r0X in r0Xs:
                for C1C2CWCY in C1C2CWCYs:
                    h=Help_Code_Merge(geom,attr,r0X,C1C2CWCY,isIntra,reGenerate)


