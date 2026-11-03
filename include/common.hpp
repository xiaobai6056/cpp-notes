#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string.h>

#include "json.hpp"

using namespace std;
using namespace cv;

enum RoadType
{
    BaseHandle = 0,
    RingHandle,
    CrossHandle,
    BridgeHandle,
    ParkingHandle,
    StartHandle,
    BurgerHandle,
    RescueHandle,
    coneHandle,
    chargeHandle,
};

struct Params
{
    bool enablePathPlanning = true;
    bool StartFlag = true;
    string modelPath = "/root/workspace/Dragon/res/model";

    bool debug = false;
    bool saveRaw = false;
    bool display = false;
    bool AIEnable = false;
    bool saveAI = false;
    bool useRemap = false;
    bool skipMainSerialCheck = true;
    bool localizationForwardOnlyMode = false;
    bool localizationSystemEnable = true;
    string localizationUdpListenIp = "0.0.0.0";
    int localizationUdpListenPort = 9001;
    string httpStreamUrl = "http://192.168.137.137:8080/ar_feed";
    string localizationUdpTargetIp = "192.168.137.137";
    int localizationUdpTargetPort = 9005;
    bool localizationPayloadLogEnable = false;
    string localizationPayloadLogPath =
        "/home/orangepi/workspace/Dragon/res/Videos/localization_forward.log";
    bool serialThreadCsvLogEnable = true;
    string serialThreadCsvLogPath =
        "/home/orangepi/workspace/Dragon/res/Videos/imu_time_align.csv";
    bool servoDebugLogEnable = false;
    string servoDebugLogPath = "servo_debug.log";
    bool imuAdDebugLogEnable = false;
    string imuAdDebugLogPath =
        "/home/orangepi/workspace/Dragon/res/Videos/imu_ad_debug.csv";
    bool localPlannerLogEnable = false;
    string localPlannerLogPath =
        "/home/orangepi/workspace/Dragon/res/Videos/local_planner.log";
    bool dynamicPathPlannerLogEnable = false;
    string dynamicPathPlannerLogPath =
        "/home/orangepi/workspace/Dragon/res/Videos/dynamic_path_planner.csv";
    bool timestampedWorkspaceSaveEnable = false;

    struct SpeedDecisionParams
    {
        bool enable = true;
        float obstacleVisibleSpeedMps = 0.85f;
        float clearSpeedMps = 1.1f;
        float virtualLookaheadSpeedMps = 0.8f;
        int clearFrames = 20;
    } speedDecision;

    struct VirtualLookaheadParams
    {
        bool enable = true;
        int lowArrowMaxCount = 4;
        int recoverArrowMinCount = 5;
        float activateDistanceM = 0.3f;
        float fixedServoOutput = 60.0f;
    } virtualLookahead;

    struct GoldPositionParams
    {
        float correctionScale = 0.85f;
    } goldPosition;

    struct GoldPathPlanningParams
    {
        float goldBiasStepM = 0.05f;
        float goldMaxBiasM = 0.20f;
        float goldCaptureCompensationM = 0.03f;
        float goldCaptureToleranceM = 0.04f;
        float goldBiasRateMps = 0.40f;
        float goldMaxNewDeltaM = 0.12f;
        int goldRecenterHoldFrames = 10;
        float goldNearPriorityDistanceM = 0.30f;
        float goldLateStartForwardM = 0.18f;
        float goldLateStartDeltaM = 0.09f;
        float goldCenterReleaseM = 0.02f;
        float pedestrianGoldDangerStripHalfWidthM = 0.20f;
        int pedestrianGoldDangerStripHoldFrames = 5;
        bool terminalPoseEnable = true;
        float terminalPosePathToleranceM = 0.05f;
        float terminalPoseActivationDistanceM = 0.50f;
        float terminalPoseKp = 3.0f;
        float terminalPoseKd = 12.0f;
        float terminalPoseDerivativeFilterAlpha = 0.6f;
        float terminalPoseDerivativeLimitDeg = 60.0f;
        float terminalPoseTargetJumpResetM = 0.12f;
        float terminalPoseServoAbs = 100.0f;
    } goldPathPlanning;

    struct servoParams
    {
        float Pbase;
        float P3base;
        float weight;
        float Ibase;
        float Dbase;
        float D3base;
        float PdP3base;
        float PdPbase;
        float PdDbase;
        float PdD3base;
        bool FuzzyEnable = false;
        bool PolyEnable = false;
        NLOHMANN_DEFINE_TYPE_INTRUSIVE(
            servoParams,
            Pbase,
            P3base,
            weight,
            Ibase,
            Dbase,
            D3base,
            PdP3base,
            PdPbase,
            PdDbase,
            PdD3base,
            FuzzyEnable,
            PolyEnable);
    } servoParams;

    struct WorldPurePursuitParams
    {
        bool enable = true;
        float aheadDistance = 0.8f;
        float wheelBase = 0.20f;
        float servoLimit = 150.0f;
        bool saveWorldView = false;
        int worldViewWidth = 640;
        int worldViewHeight = 480;
        float worldViewForwardMeters = 2.5f;
        float worldViewSideMeters = 1.2f;
        NLOHMANN_DEFINE_TYPE_INTRUSIVE(
            WorldPurePursuitParams,
            enable,
            aheadDistance,
            wheelBase,
            servoLimit,
            saveWorldView,
            worldViewWidth,
            worldViewHeight,
            worldViewForwardMeters,
            worldViewSideMeters);
    } worldPurePursuitParams;

    struct DirectErrorPIDParams
    {
        float lateralErrorFilterAlpha = 0.7;
        NLOHMANN_DEFINE_TYPE_INTRUSIVE(DirectErrorPIDParams, lateralErrorFilterAlpha);
    } directErrorPIDParams;

    struct TrendControlParams
    {
        float errorGain = 0.25f;
        float trendGain = 0.15f;
        float dErrGain = 0.01f;
        float trendMax = 80.0f;
        float dErrMax = 800.0f;
        float pixelToServo = 2.5f;
        float servoOutputLimit = 150.0f;
        float pathErrorYMin = 240.0f;
        float pathErrorYMax = 480.0f;
        NLOHMANN_DEFINE_TYPE_INTRUSIVE(
            TrendControlParams,
            errorGain,
            trendGain,
            dErrGain,
            trendMax,
            dErrMax,
            pixelToServo,
            servoOutputLimit,
            pathErrorYMin,
            pathErrorYMax);
    } trendControlParams;

    struct PEATHSearchParams
    {
        float ROAD_WIDTH = 0.30;
        float pixel_per_meter = 222.5;
        float kl = 1.17;
        float bl = 129;
        float kr = -1.26;
        float br = 523;
        int edgeMaximum = 200;
        NLOHMANN_DEFINE_TYPE_INTRUSIVE(
            PEATHSearchParams, ROAD_WIDTH, pixel_per_meter, kl, bl, kr, br, edgeMaximum);
    } pathSearchParams;

    struct PathErrorParams
    {
        float IMAGE_CENTER_X = 320.0;
        float PATH_ERROR_MAX = 150.0;
        NLOHMANN_DEFINE_TYPE_INTRUSIVE(PathErrorParams, IMAGE_CENTER_X, PATH_ERROR_MAX);
    } pathErrorParams;

    struct RoadSignForkParams
    {
        bool enable = true;
        bool secondOppositeOnlyEnable = false;
        int signAreaThreshold = 1500;
        int signStableFrames = 2;
        int signLostHoldFrames = 6;
        int decisionStopFrames = 3;
        int decisionTimeoutMs = 10000;
        int defaultDecision = 1;
        float stopSlowdownSpeedMps = 0.45f;
        int stopSlowdownStableFrames = 1;
        int stopSlowdownHoldFrames = 6;
        int stopSlowdownAreaThreshold = 8000;
        int armTimeoutFrames = 90;
        int activeLostExitFrames = 20;
        int forkStableFrames = 2;
        int forkExitFrames = 4;
        int cooldownFrames = 20;
        int postPassCooldownMs = 3000;
        bool poseAdjustEnable = true;
        float poseVirtualRadiusM = 0.50f;
        float poseCenterFilterAlpha = 0.65f;
        float poseCenterDeadbandPx = 10.0f;
        float poseKp = 3.8f;
        float poseKd = 22.0f;
        float poseDerivativeFilterAlpha = 0.15f;
        float poseDerivativeLimitDeg = 300.0f;
        float poseTargetJumpResetM = 0.15f;
        float poseServoAbs = 80.0f;
        float poseSafetyBlockDistanceM = 0.80f;
        bool decisionServoHoldEnable = true;
        float decisionServoLeftOutput = 30.0f;
        float decisionServoRightOutput = 60.0f;
        int decisionServoHoldMs = 300;
        NLOHMANN_DEFINE_TYPE_INTRUSIVE(RoadSignForkParams,
                                       enable,
                                       signStableFrames,
                                       signLostHoldFrames,
                                       armTimeoutFrames,
                                       forkStableFrames,
                                       forkExitFrames,
                                       cooldownFrames);
    } roadSignFork;

    struct SpeedLightParams
    {
        bool enable = true;
        bool stateMachineEnable = false;
        float limitSpeed = 0.30f;
        float stopSpeed = 0.0f;
        int speedLimitStableFrames = 2;
        int lightStableFrames = 2;
        int signLostHoldFrames = 6;
        int greenReleaseFrames = 2;
        int passLightExitFrames = 6;
        int cooldownFrames = 20;
        int lightTriggerMinBottomY = 220;
        int lightTriggerMinHeight = 18;
        NLOHMANN_DEFINE_TYPE_INTRUSIVE(SpeedLightParams,
                                       enable,
                                       limitSpeed,
                                       stopSpeed,
                                       speedLimitStableFrames,
                                       lightStableFrames,
                                       signLostHoldFrames,
                                       greenReleaseFrames,
                                       passLightExitFrames,
                                       cooldownFrames,
                                       lightTriggerMinBottomY,
                                       lightTriggerMinHeight);
    } speedLight;

    struct StopSignStopParams
    {
        bool enable = true;
        int stableFrames = 2;
        int stopAfterLostMs = 2000;
        int startProtectMs = 10000;
        NLOHMANN_DEFINE_TYPE_INTRUSIVE(StopSignStopParams,
                                       enable,
                                       stableFrames,
                                       stopAfterLostMs,
                                       startProtectMs);
    } stopSignStop;

    struct ObstacleAvoidanceParams
    {
        bool enable = true;
        int confirmFrames = 3;
        int exitLostFrames = 10;
        float prepareDistanceM = 1.0f;
        float activateDistanceM = 0.8f;
        float vehicleLengthM = 0.32f;
        float vehicleWidthM = 0.22f;
        float vehicleFrontM = 0.17f;
        float vehicleRearM = 0.15f;
        bool forceLeftAvoidance = false;
        int vehicleAvoidanceRecoverHoldFrames = 5;
        int vehicleRecoverHoldMaxLifetimeFrames = 45;
        float avoidOffsetM = 0.15f;
        float vehicleAvoidanceEdgeClearanceM = 0.18f;
        float vehicleAvoidanceMaxOffsetM = 0.55f;
        float vehicleCurveAvoidanceExtraOffsetM = 0.05f;
        float vehicleAvoidanceOppositeSteerThresholdDeg = 5.0f;
        int vehicleAvoidanceOppositeSteerHoldFrames = 8;
        float offsetRateMps = 0.45f;
        float offsetZeroEpsilonM = 0.001f;
        float pedestrianDangerCorridorNearWidthM = 0.70f;
        float pedestrianDangerCorridorFarWidthM = 0.22f;
        float pedestrianDangerCorridorLengthM = 0.70f;
        float pedestrianDangerCorridorSampleStepM = 0.03f;
        float pedestrianServoStraightDeadband = 10.0f;
        float pedestrianServoRadiusMaxAbs = 140.0f;
        float pedestrianDangerCorridorServoSmoothAlpha = 0.25f;
        float pedestrianDangerCorridorServoMaxStep = 10.0f;
        bool pedestrianSlowEnable = true;
        float pedestrianSlowDistanceM = 0.8f;
        float pedestrianSlowSpeedMps = 0.25f;
        int pedestrianMotionWindowFrames = 4;
        float pedestrianMotionTrendThresholdM = 0.010f;
        float pedestrianMotionMaxJumpM = 0.15f;
        int pedestrianSlowAwayConfirmFrames = 2;
        int pedestrianSlowObservationFrames = 6;
        int pedestrianSlowLostFrames = 3;
        int pedestrianStopClearFrames = 3;
        int pedestrianReleaseLostGraceFrames = 2;
        float pedestrianExitBoundaryMarginM = 0.05f;
        float pedestrianStopCarSideThresholdM = 0.08f;
        int startupLaunchProtectFrames = 30;
        float pedestrianStoppedLaunchSpeedMps = 1.3f;
        float pedestrianSlowLaunchSpeedMps = 1.0f;
        int pathSmoothWindow = 5;
        float avoidLookaheadDistanceM = 0.5f;
    } obstacleAvoidance;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(
        Params,
        StartFlag,
        modelPath,
        servoParams,
        worldPurePursuitParams,
        directErrorPIDParams,
        trendControlParams,
        roadSignFork,
        speedLight,
        stopSignStop,
        AIEnable,
        pathSearchParams,
        pathErrorParams,
        debug,
        saveRaw,
        display,
        saveAI,
        useRemap,
        skipMainSerialCheck,
        serialThreadCsvLogEnable);
};

#define COLSIMAGE 320
#define ROWSIMAGE 240
#define COLSIMAGEIPM 320
#define ROWSIMAGEIPM 400
#define PWMSERVOMAX 1900
#define PWMSERVOMID 1500
#define PWMSERVOMIN 1100

#define LABEL_ARROW "arrow"
#define LABEL_PEOPLE "people"
#define LABEL_CAR "car"
#define LABEL_GOLD "gold"

#define TYPE_ARROW 0
#define TYPE_PEOPLE 1
#define TYPE_CAR 2
#define TYPE_GOLD 3

#ifndef TYPE_DOOR
#define TYPE_DOOR 4
#define TYPE_ROAD_SIGN 5
#define TYPE_SPEED_LIMIT_SIGN 6
#define TYPE_RED_LIGHT 7
#define TYPE_YELLOW_LIGHT 8
#define TYPE_GREEN_LIGHT 9
#define TYPE_STOP_SIGN 10
#endif

Params paramsInit(string configPath,int *flag);
