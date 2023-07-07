import pandas as pd
import numpy as np
from scipy.integrate import quad

import matplotlib.pyplot as plt


class Distortion:
    def __init__(self, dataframe):
        self.initialDF = dataframe

        probeNumber = self.initialDF['Probe']
        span = self.initialDF['Span']
        theta = self.initialDF['Theta (radians)']
        pressure = self.initialDF['Pressure (kPa)']
        #TODO set up try/except cases for different possible column names

        self.df = pd.concat([probeNumber, span, theta, pressure], axis=1)
        self.df.columns = ['Number', 'Span', 'Theta', 'Pressure']
        self.df = self.df.sort_values(['Theta', 'Span'])
        self.df = self.df.reset_index(drop=True)

        self.hubRadius = 0
        self.casingRadius = 1

    def getDF(self):
        return self.df

    def ARP1420(self):

        spans = self.df['Span'].drop_duplicates()
        circumIntenseOutList = []
        circumExtentOutList = []
        MPROutList = []
        radialIntenseOutList = []
        for i in spans:
            PAV = self.ARP1420RingPAV(i)
            thetas, pressures = self.getRingData(i)
            thetas = np.concatenate((thetas, [thetas[0] + 2 * np.pi]), axis=0)
            pressures = np.concatenate((pressures, [pressures[0]]), axis=0)
            f_interp = lambda xx: np.interp(xx, thetas, pressures)

            h = 1000
            values = np.linspace(0, 2 * np.pi, h)
            transitions = []  # list of angular locations where pressure passes through PAV
            aboveToBelowFirst = None
            for j, value in enumerate(values):
                if j < h - 1:
                    if (f_interp(value) > PAV) & (f_interp(values[j + 1]) < PAV):
                        transitions = transitions + [value]
                        if len(transitions) == 1:
                            aboveToBelowFirst = True
                    if (f_interp(value) < PAV) & (f_interp(values[j + 1]) > PAV):
                        transitions = transitions + [value]
                        if len(transitions) == 1:
                            aboveToBelowFirst = False

            extentList = []
            for k, value in enumerate(transitions):
                if k < len(transitions) - 1:
                    newVal1 = transitions[k + 1] - value
                else:
                    newVal1 = transitions[0] - value
                while newVal1 < 0:
                    newVal1 = newVal1 + 2 * np.pi
                while newVal1 > 2 * np.pi:
                    newVal1 = newVal1 - 2 * np.pi
                extentList = extentList + [newVal1]
            if aboveToBelowFirst:
                lowPExtent = extentList[0::2]
                highPExtent = extentList[1::2]
            else:
                lowPExtent = extentList[1::2]
                highPExtent = extentList[0::2]

            extent = np.sum(lowPExtent)

            thetaMin = 25 * (np.pi / 180)
            PAVLOW = None
            MPR = None
            if len(transitions) == 2:
                #One-per-rev pattern
                MPR = 1
                if aboveToBelowFirst:
                    [integral, error] = quad(f_interp, transitions[0], transitions[1], points=thetas)
                    PAVLOW = (1/extent)*integral
                else:
                    [integral1, error] = quad(f_interp, 0, transitions[0], points=thetas)
                    [integral2, error] = quad(f_interp, transitions[1], 2*np.pi, points=thetas)
                    PAVLOW = (1/extent)*(integral1+integral2)
                intensity = (PAV-PAVLOW)/PAV
                MPROutList = MPROutList + [MPR]
                circumIntenseOutList = circumIntenseOutList + [intensity]

            elif min(highPExtent) < thetaMin:
                #Multiple-per-rev pattern, high pressure extent < thetaMin
                MPR = 1
                integralSum = 0
                if aboveToBelowFirst:
                    for i in range(0, len(transitions) - 1, 2):
                        lowBound = transitions[i]
                        highBound = transitions[i + 1]
                        [integral, error] = quad(f_interp, lowBound, highBound, points=thetas)
                        integralSum = integralSum + integral
                    PAVLOW = (1/extent)*integralSum
                else:
                    [integral1, error] = quad(f_interp, 0, transitions[0], points=thetas)
                    [integral2, error] = quad(f_interp, transitions[-1], 2*np.pi, points=thetas)
                    integralSum = integral1+integral2
                    for i in range(1, len(transitions) - 2, 2):
                        lowBound = transitions[i]
                        highBound = transitions[i + 1]
                        [integral, error] = quad(f_interp, lowBound, highBound, points=thetas)
                        integralSum = integralSum + integral
                    PAVLOW = (1 / extent) * integralSum
                intensity = (PAV - PAVLOW) / PAV
                MPROutList = MPROutList + [MPR]
                circumIntenseOutList = circumIntenseOutList + [intensity]
            else:
                #Multiple-per-rev pattern, high pressure extent > thetaMin
                intensityList = []
                extentWeightedIntensityList = []
                extents = []

                if aboveToBelowFirst:
                    for i in range(0, len(transitions) - 1, 2):
                        lowBound = transitions[i]
                        highBound = transitions[i + 1]
                        [integral, error] = quad(f_interp, lowBound, highBound, points=thetas)
                        PAVLOW = (1/lowPExtent[int(i/2)])*integral
                        intensity = (PAV-PAVLOW)/PAV
                        intensityList = intensityList + [intensity]
                        extentWeightedIntensityList = extentWeightedIntensityList + [intensity*lowPExtent[int(i/2)]]
                        extents = extents + [lowPExtent[int(i/2)]]
                else:
                    for i in range(1, len(transitions) - 2, 2):
                        lowBound = transitions[i]
                        highBound = transitions[i + 1]
                        [integral, error] = quad(f_interp, lowBound, highBound, points=thetas)
                        integralSum = integralSum + integral
                        PAVLOW = (1 / lowPExtent[int(i%2+1)]) * integral
                        intensity = (PAV - PAVLOW) / PAV
                        intensityList = intensityList + [intensity]
                        extentWeightedIntensityList = extentWeightedIntensityList + [intensity * lowPExtent[int(i%2+1)]]
                        extents = extents + [lowPExtent[int(i%2+1)]]
                    [integral1, error] = quad(f_interp, 0, transitions[0], points=thetas)
                    [integral2, error] = quad(f_interp, transitions[-1], 2 * np.pi, points=thetas)
                    integralSum = integral1 + integral2
                    PAVLOW = (1 / lowPExtent[-1]) * integralSum
                    intensity = (PAV - PAVLOW) / PAV
                    intensityList = intensityList + [intensity]
                    extentWeightedIntensityList = extentWeightedIntensityList + [intensity * lowPExtent[-1]]
                    extents = extents + [lowPExtent[-1]]
                    print(intensityList)
                    print(extentWeightedIntensityList)

                index = pd.Series(extentWeightedIntensityList).idxmax()
                intensity = intensityList[index]
                extent = extents[index]
                MPR = sum(extentWeightedIntensityList)/extentWeightedIntensityList[index]
                MPROutList = MPROutList + [MPR]
                circumIntenseOutList = circumIntenseOutList + [intensity]

            circumExtentOutList = circumExtentOutList + [extent]

            PFAV = self.ARP1420PFAVEqualRingArea()
            radialIntensity = (PFAV-PAV)/PFAV
            radialIntenseOutList = radialIntenseOutList + [radialIntensity]

        circumExtentOutList = [i*(180/np.pi) for i in circumExtentOutList]
        circumIntense = pd.DataFrame({'circumIntense': circumIntenseOutList})
        circumExtent = pd.DataFrame({'circumExtent': circumExtentOutList})
        MPR = pd.DataFrame({'MPR': MPROutList})
        radialIntense = pd.DataFrame({'Radial Intense': radialIntenseOutList})
        output = pd.concat([spans, circumIntense, circumExtent, MPR, radialIntense], axis=1)
        output.columns = ['Span/ Ring #', 'Circumferential Intensity', 'Circumferential Extent (deg)',
                          'Multiple Per Rev', 'Radial Intensity']
        return output

    def ARP1420RingPAV(self, span):
        thetas, pressures = self.getRingData(span)
        thetas = np.concatenate((thetas, [thetas[0] + 2 * np.pi]), axis=0)
        pressures = np.concatenate((pressures, [pressures[0]]), axis=0)
        f_interp = lambda xx: np.interp(xx, thetas, pressures)
        [integral, error] = quad(f_interp, 0, 2 * np.pi, points=thetas)
        ringAvg = 1 / (2 * np.pi) * integral
        return ringAvg


    def ARP1420PFAVEqualRingArea(self):
        spans = self.df['Span'].drop_duplicates()
        N = spans.size
        sum = 0
        for i in spans:
            sum = sum + self.ARP1420RingPAV(i)
        PFAV = (1 / N) * sum
        return PFAV

    def getRingData(self, span):
        data = self.df.loc[self.df['Span'] == span]
        thetas = data['Theta'].to_numpy()
        pressures = data['Pressure'].to_numpy()
        return thetas, pressures

    def pDeltaPavg1(self):
        """Returns a simple distortion index

        :math:`\\frac{\\Delta P_{max-min}}{\\bar{P}}=\\frac{P_{max}-P_{min}}{P_{avg}}`

        :math:`P_{max}` = Maximum inlet total pressure

        :math:`P_{min}` = Minimum inlet total pressure

        :math:`P_{avg}` = Average inlet total pressure

        :return: Calculated Distortion Index
        :rtype: float
        """
        pressures = self.df['Pressure'].to_numpy()
        pMin = np.min(pressures)
        pMax = np.max(pressures)
        pAvg = np.average(pressures)
        index = (pMax - pMin) / pAvg
        return index

    def pDeltaPavg2(self):
        """Returns a simple distortion index

        :math:`\\frac{\\Delta P_{avg-min}}{\\bar{P}}=\\frac{P_{avg}-P_{min}}{P_{avg}}`

        :math:`P_{min}` = Minimum inlet total pressure

        :math:`P_{avg}` = Average inlet total pressure


        :return: Calculated Distortion Index
        :rtype: float
        """
        pressures = self.df['Pressure'].to_numpy()
        pMin = np.min(pressures)
        pAvg = np.average(pressures)

        index = (pAvg - pMin) / pAvg
        return index

    def RRCriticalAngle(self):
        """Returns the Rolls-Royce critical angle index WORK IN PROGRESS


        :math:`\\Delta P(\\Theta critical)/\\bar{P} = \\frac{P_{avg}-P_{min},\\Theta^{-}_{c},avg}{P_{avg}}`

        :math:`DC(\\Theta critical)/\\bar{P} = \\frac{P_{avg}-P_{min},\\Theta^{-}_{c},avg}{q_{avg}}`

        :math:`P_{avg}` = The area-weighted mean total pressure over the engine inlet

        :math:`P_{min},\\Theta^{-}_{c},avg` = The minimum area-weighted mean total pressure for a sector whose circumferential extent is '\\Theta' critical

        :math:`q_{avg}` = The area-weighted average velocity head over the engine inlet

        :return: Calculated Distortion Index
        :rtype: float
        """

    def NAPCKTheta(self):
        """Returns the Naval Air Propulsion Center KTheta index WORK IN PROGRESS

        :math:`K\\Theta = \\frac{\\frac{\\Theta^{-}}{2\\pi}[\\sqrt{q/P}]_{ref}}{\\sqrt{\\frac{q}{P}/\\frac{\\bar{q}}{\\bar{P}}}}`

        :math:`\\Theta^{-}` = Circumferential extent of the total pressure region less than the plane average total pressure

        :math:`P` = Average inlet total pressure within the low pressure region

        :math:`\\bar{P}` = Average inlet total pressure

        :math:`q` = Average dynamic pressure in low pressure region

        :math:`\\bar{q}` = Average inlet dynamic pressure

        :return: Calculated Distortion Index
        :rtype: float
        """

    def AVCOLycomingDI(self):
        """Returns the AVCO Lycoming DI index WORK IN PROGRESS

        :math:`DI = (\\frac{P_{avg}-P_{low\\:avg}}{P_{avg}})\\sqrt{\\overline{M*E*R}}`

        :math:`P_{avg}` = Area-weighted average total pressure

        :math:`P_{low\\:avg}` = Area-weighted total pressure in regions where P is less than :math:`P_{avg}`

        :math:`M` = Magnitude or shape factor = :math:`6.0(P_{avg}-P_{low\\:avg})/(P_{avg}-P_{low\\:min})`

        :math:`P_{low\\:min}` = minimum total pressure level

        :math:`E` = Extent of distorted region = :math:`2.0(A_{L})/A_{tot}`

        :math:`A_{L}` = Area over which the total pressure is less than :math:`P_{avg}`

        :math:`A_{tot}` = total annulus area

        :math:`R` = Radial distortion sensitivity = maximum of :math:`2.0(A_{L,hub}/A_{L})` or :math:`2.0(A_{L,tip}/A_{L})`

        :math:`A_{L,hub}` = Area extent of low pressure regions which fall in the inner 50% annulus area

        :math:`A_{L,tip}` = Area extent of low pressure regions which fall in the outer 50% annulus area

        :return: Calculated Distortion Index
        :rtype: float
        """

    def get_areaWeightedAverage(self):
        spans = self.df['Span'].drop_duplicates().to_numpy()
        thetas = self.df['Theta'].drop_duplicates().to_numpy()
        thetaDifferences = np.diff(thetas) / 2  # Half the angle between each rake
        thetaDifferences = np.concatenate((thetaDifferences, [((thetas[0] + (2 * np.pi - thetas[-1])) / 2)]), axis=0)
        # Creating list of differences in angle between adjacent rakes, divided by 2
        sectionEdgeAngles = thetas + thetaDifferences  # Creating list of angles midway between each rake - rake 
        # bisectors
        sectionDiff = np.diff(sectionEdgeAngles)  # Finding the difference in angle between rake bisectors
        sectorAngles = sectionEdgeAngles[0] - sectionEdgeAngles[-1] % (
                2 * np.pi)
        while sectorAngles < 0:
            sectorAngles = sectorAngles + 2 * np.pi
        while sectorAngles > 2 * np.pi:
            sectorAngles = sectorAngles - 2 * np.pi
        # Assigning first annular sector angle since it wraps around array
        sectorAngles = np.concatenate(([sectorAngles], sectionDiff), axis=0)  # Adding rest of sector angles
        # TODO There's almost certainly a better way of doing some of this angle math ^

        spanDifferences = np.diff(spans) / 2  # Half the distance between each probe radially
        circleRadii = np.concatenate(([self.hubRadius], spans[0:-1] + spanDifferences, [self.casingRadius]),
                                     axis=0)  # Combining
        # hubRadius, casingRadius, and midpoints of probes to list relevant circle radii
        ringAreas = []
        for i, radius in enumerate(circleRadii):
            if i < len(circleRadii) - 1:
                area = np.pi * (pow(circleRadii[i + 1], 2) - pow(circleRadii[i], 2))
                ringAreas = ringAreas + [area]
        # List of the relevant ring areas from center outwards
        effectiveArea = sum(ringAreas)  # Sum of rings to find effective area (area between casing and hub)
        weights = []
        pressures = []
        for i, pressure in enumerate(self.df['Pressure'].to_numpy()):
            ringNumber = (i + 1) % len(ringAreas) - 1
            angleNumber = (i) // len(ringAreas)
            weights = weights + [(ringAreas[ringNumber] * (sectorAngles[angleNumber] / (2 * np.pi))) / effectiveArea]
            pressures = pressures + [pressure]

        # print(np.sum(weights))
        weightedAverage = np.dot(weights, pressures)
        # Dot product of area weights and sector pressures

        return weightedAverage

    def get_radialAverage(self):
        thetasRaw = self.df['Theta'].to_numpy()  # Raw list of thetas from sorted csv (has repeats)
        diff_list = np.diff(thetasRaw)  # Finding differences between adjacent theta values
        I = np.nonzero(diff_list)  # Determining the location of first nonzero element
        probesOnRake = 1 + I[0][0]  # Number of probes on each rake
        thetas = thetasRaw[::probesOnRake]  # Creating new thetas array with no repeats

        pressures = self.df['Pressure'].to_numpy()
        radialAvg = pd.DataFrame(columns=['Theta', 'Average'])
        for i in range(len(thetas)):
            sum = 0
            for j in range(probesOnRake):
                sum = sum + pressures[i * probesOnRake + j]

            newRow = pd.DataFrame(
                {'Theta': [thetas[i]],
                 'Average': [sum / probesOnRake]
                 })
            radialAvg = pd.concat([radialAvg, newRow])
        return radialAvg

    def get_circumferentialAverage(self):
        thetasRaw = self.df['Theta'].to_numpy()  # Raw list of thetas from sorted csv (has repeats)
        diff_list = np.diff(thetasRaw)  # Finding differences between adjacent theta values
        I = np.nonzero(diff_list)  # Determining the location of first nonzero element
        probesOnRake = 1 + I[0][0]  # Number of probes on each rake
        thetas = thetasRaw[::probesOnRake]  # Creating new thetas array with no repeats

        spans = self.df['Span'].to_numpy()
        spans = spans[0:probesOnRake]
        pressures = self.df['Pressure'].to_numpy()
        circumferentialAvg = pd.DataFrame(columns=['Span', 'Average'])
        for i in range(probesOnRake):
            sum = 0
            for j in range(len(thetas)):
                sum = sum + pressures[j * probesOnRake + i]

            newRow = pd.DataFrame(
                {'Span': [spans[i]],
                 'Average': [sum / probesOnRake]
                 })
            circumferentialAvg = pd.concat([circumferentialAvg, newRow])
        return circumferentialAvg
