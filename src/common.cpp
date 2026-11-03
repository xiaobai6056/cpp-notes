#include "../include/common.hpp"

#include <cmath>

Params paramsInit(string configPath, int *flag)
{
    Params params;

    const char *temp = configPath.c_str();
    std::ifstream config_is(temp, ios::in);
    if (!config_is.good())
    {
        std::cout << "Error: Params file path:[" << configPath
                  << "] not find .\n";
        *flag = 0;
        return params;
    }

    nlohmann::json js_value;
    config_is >> js_value;

    try
    {
        params = js_value.get<Params>();

        // Optional fields are parsed manually to preserve compatibility with
        // older config files that may not carry the new keys yet.
        if (js_value.contains("roadSignFork") &&
            js_value["roadSignFork"].is_object()) {
            const auto& road_sign = js_value["roadSignFork"];
            if (road_sign.contains("secondOppositeOnlyEnable") &&
                road_sign["secondOppositeOnlyEnable"].is_boolean()) {
                params.roadSignFork.secondOppositeOnlyEnable =
                    road_sign["secondOppositeOnlyEnable"].get<bool>();
            }
            if (road_sign.contains("signAreaThreshold") &&
                road_sign["signAreaThreshold"].is_number_integer()) {
                params.roadSignFork.signAreaThreshold =
                    road_sign["signAreaThreshold"].get<int>();
            }
            if (road_sign.contains("decisionStopFrames") &&
                road_sign["decisionStopFrames"].is_number_integer()) {
                params.roadSignFork.decisionStopFrames =
                    road_sign["decisionStopFrames"].get<int>();
            }
            if (road_sign.contains("decisionTimeoutMs") &&
                road_sign["decisionTimeoutMs"].is_number_integer()) {
                params.roadSignFork.decisionTimeoutMs =
                    road_sign["decisionTimeoutMs"].get<int>();
            }
            if (road_sign.contains("defaultDecision") &&
                road_sign["defaultDecision"].is_number_integer()) {
                params.roadSignFork.defaultDecision =
                    road_sign["defaultDecision"].get<int>();
            }
            if (road_sign.contains("stopSlowdownSpeedMps") &&
                road_sign["stopSlowdownSpeedMps"].is_number()) {
                params.roadSignFork.stopSlowdownSpeedMps =
                    road_sign["stopSlowdownSpeedMps"].get<float>();
            }
            if (road_sign.contains("stopSlowdownStableFrames") &&
                road_sign["stopSlowdownStableFrames"].is_number_integer()) {
                params.roadSignFork.stopSlowdownStableFrames =
                    road_sign["stopSlowdownStableFrames"].get<int>();
            }
            if (road_sign.contains("stopSlowdownHoldFrames") &&
                road_sign["stopSlowdownHoldFrames"].is_number_integer()) {
                params.roadSignFork.stopSlowdownHoldFrames =
                    road_sign["stopSlowdownHoldFrames"].get<int>();
            }
            if (road_sign.contains("stopSlowdownAreaThreshold") &&
                road_sign["stopSlowdownAreaThreshold"].is_number_integer()) {
                params.roadSignFork.stopSlowdownAreaThreshold =
                    road_sign["stopSlowdownAreaThreshold"].get<int>();
            }
            if (road_sign.contains("postPassCooldownMs") &&
                road_sign["postPassCooldownMs"].is_number_integer()) {
                params.roadSignFork.postPassCooldownMs =
                    road_sign["postPassCooldownMs"].get<int>();
            }
            if (road_sign.contains("poseAdjustEnable") &&
                road_sign["poseAdjustEnable"].is_boolean()) {
                params.roadSignFork.poseAdjustEnable =
                    road_sign["poseAdjustEnable"].get<bool>();
            }
            if (road_sign.contains("poseVirtualRadiusM") &&
                road_sign["poseVirtualRadiusM"].is_number()) {
                params.roadSignFork.poseVirtualRadiusM =
                    road_sign["poseVirtualRadiusM"].get<float>();
            }
            if (road_sign.contains("poseCenterFilterAlpha") &&
                road_sign["poseCenterFilterAlpha"].is_number()) {
                params.roadSignFork.poseCenterFilterAlpha =
                    road_sign["poseCenterFilterAlpha"].get<float>();
            }
            if (road_sign.contains("poseCenterDeadbandPx") &&
                road_sign["poseCenterDeadbandPx"].is_number()) {
                params.roadSignFork.poseCenterDeadbandPx =
                    road_sign["poseCenterDeadbandPx"].get<float>();
            }
            if (road_sign.contains("poseKp") && road_sign["poseKp"].is_number()) {
                params.roadSignFork.poseKp = road_sign["poseKp"].get<float>();
            }
            if (road_sign.contains("poseKd") && road_sign["poseKd"].is_number()) {
                params.roadSignFork.poseKd = road_sign["poseKd"].get<float>();
            }
            if (road_sign.contains("poseDerivativeFilterAlpha") &&
                road_sign["poseDerivativeFilterAlpha"].is_number()) {
                params.roadSignFork.poseDerivativeFilterAlpha =
                    road_sign["poseDerivativeFilterAlpha"].get<float>();
            }
            if (road_sign.contains("poseDerivativeLimitDeg") &&
                road_sign["poseDerivativeLimitDeg"].is_number()) {
                params.roadSignFork.poseDerivativeLimitDeg =
                    road_sign["poseDerivativeLimitDeg"].get<float>();
            }
            if (road_sign.contains("poseTargetJumpResetM") &&
                road_sign["poseTargetJumpResetM"].is_number()) {
                params.roadSignFork.poseTargetJumpResetM =
                    road_sign["poseTargetJumpResetM"].get<float>();
            }
            if (road_sign.contains("poseServoAbs") &&
                road_sign["poseServoAbs"].is_number()) {
                params.roadSignFork.poseServoAbs =
                    road_sign["poseServoAbs"].get<float>();
            }
            if (road_sign.contains("poseSafetyBlockDistanceM") &&
                road_sign["poseSafetyBlockDistanceM"].is_number()) {
                params.roadSignFork.poseSafetyBlockDistanceM =
                    road_sign["poseSafetyBlockDistanceM"].get<float>();
            }
            if (road_sign.contains("decisionServoHoldEnable") &&
                road_sign["decisionServoHoldEnable"].is_boolean()) {
                params.roadSignFork.decisionServoHoldEnable =
                    road_sign["decisionServoHoldEnable"].get<bool>();
            }
            if (road_sign.contains("decisionServoLeftOutput") &&
                road_sign["decisionServoLeftOutput"].is_number()) {
                params.roadSignFork.decisionServoLeftOutput =
                    road_sign["decisionServoLeftOutput"].get<float>();
            }
            if (road_sign.contains("decisionServoRightOutput") &&
                road_sign["decisionServoRightOutput"].is_number()) {
                params.roadSignFork.decisionServoRightOutput =
                    road_sign["decisionServoRightOutput"].get<float>();
            }
            if (road_sign.contains("decisionServoHoldMs") &&
                road_sign["decisionServoHoldMs"].is_number_integer()) {
                params.roadSignFork.decisionServoHoldMs =
                    road_sign["decisionServoHoldMs"].get<int>();
            }
            if (road_sign.contains("activeLostExitFrames") &&
                road_sign["activeLostExitFrames"].is_number_integer()) {
                params.roadSignFork.activeLostExitFrames =
                    road_sign["activeLostExitFrames"].get<int>();
            }
            params.roadSignFork.stopSlowdownAreaThreshold = std::max(
                0,
                std::min(params.roadSignFork.stopSlowdownAreaThreshold,
                         std::max(0, params.roadSignFork.signAreaThreshold - 1)));
            params.roadSignFork.poseVirtualRadiusM =
                std::max(0.05f, params.roadSignFork.poseVirtualRadiusM);
            params.roadSignFork.poseCenterFilterAlpha =
                std::clamp(params.roadSignFork.poseCenterFilterAlpha, 0.0f, 1.0f);
            params.roadSignFork.poseCenterDeadbandPx =
                std::max(0.0f, params.roadSignFork.poseCenterDeadbandPx);
            params.roadSignFork.poseDerivativeFilterAlpha =
                std::clamp(params.roadSignFork.poseDerivativeFilterAlpha, 0.0f, 1.0f);
            params.roadSignFork.poseDerivativeLimitDeg =
                std::max(0.0f, params.roadSignFork.poseDerivativeLimitDeg);
            params.roadSignFork.poseTargetJumpResetM =
                std::max(0.0f, params.roadSignFork.poseTargetJumpResetM);
            params.roadSignFork.poseServoAbs =
                std::max(0.0f, params.roadSignFork.poseServoAbs);
            params.roadSignFork.poseSafetyBlockDistanceM =
                std::max(0.0f, params.roadSignFork.poseSafetyBlockDistanceM);
            params.roadSignFork.activeLostExitFrames =
                std::max(1, params.roadSignFork.activeLostExitFrames);
        }
        if (js_value.contains("httpStreamUrl") && js_value["httpStreamUrl"].is_string()) {
            params.httpStreamUrl = js_value["httpStreamUrl"].get<std::string>();
        }
        if (js_value.contains("timestampedWorkspaceSaveEnable") &&
            js_value["timestampedWorkspaceSaveEnable"].is_boolean()) {
            params.timestampedWorkspaceSaveEnable =
                js_value["timestampedWorkspaceSaveEnable"].get<bool>();
        }
        if (js_value.contains("localPlannerLogEnable") &&
            js_value["localPlannerLogEnable"].is_boolean()) {
            params.localPlannerLogEnable = js_value["localPlannerLogEnable"].get<bool>();
        }
        if (js_value.contains("localPlannerLogPath") &&
            js_value["localPlannerLogPath"].is_string()) {
            params.localPlannerLogPath = js_value["localPlannerLogPath"].get<std::string>();
        }
        if (js_value.contains("dynamicPathPlannerLogEnable") &&
            js_value["dynamicPathPlannerLogEnable"].is_boolean()) {
            params.dynamicPathPlannerLogEnable =
                js_value["dynamicPathPlannerLogEnable"].get<bool>();
        }
        if (js_value.contains("dynamicPathPlannerLogPath") &&
            js_value["dynamicPathPlannerLogPath"].is_string()) {
            params.dynamicPathPlannerLogPath =
                js_value["dynamicPathPlannerLogPath"].get<std::string>();
        }
        if (js_value.contains("localizationSystemEnable") &&
            js_value["localizationSystemEnable"].is_boolean()) {
            params.localizationSystemEnable = js_value["localizationSystemEnable"].get<bool>();
        }
        if (js_value.contains("localizationUdpListenIp") &&
            js_value["localizationUdpListenIp"].is_string()) {
            params.localizationUdpListenIp =
                js_value["localizationUdpListenIp"].get<std::string>();
        }
        if (js_value.contains("localizationUdpListenPort") &&
            js_value["localizationUdpListenPort"].is_number_integer()) {
            params.localizationUdpListenPort =
                js_value["localizationUdpListenPort"].get<int>();
        }
        if (js_value.contains("localizationUdpTargetIp") &&
            js_value["localizationUdpTargetIp"].is_string()) {
            params.localizationUdpTargetIp =
                js_value["localizationUdpTargetIp"].get<std::string>();
        }
        if (js_value.contains("localizationUdpTargetPort") &&
            js_value["localizationUdpTargetPort"].is_number_integer()) {
            params.localizationUdpTargetPort =
                js_value["localizationUdpTargetPort"].get<int>();
        }
        if (js_value.contains("localizationPayloadLogEnable") &&
            js_value["localizationPayloadLogEnable"].is_boolean()) {
            params.localizationPayloadLogEnable =
                js_value["localizationPayloadLogEnable"].get<bool>();
        }
        if (js_value.contains("localizationPayloadLogPath") &&
            js_value["localizationPayloadLogPath"].is_string()) {
            params.localizationPayloadLogPath =
                js_value["localizationPayloadLogPath"].get<std::string>();
        }
        if (js_value.contains("localizationForwardOnlyMode") &&
            js_value["localizationForwardOnlyMode"].is_boolean()) {
            params.localizationForwardOnlyMode =
                js_value["localizationForwardOnlyMode"].get<bool>();
        }
        if (js_value.contains("enable_path_planning") &&
            js_value["enable_path_planning"].is_boolean()) {
            params.enablePathPlanning = js_value["enable_path_planning"].get<bool>();
        }
        if (js_value.contains("serialThreadCsvLogEnable") &&
            js_value["serialThreadCsvLogEnable"].is_boolean()) {
            params.serialThreadCsvLogEnable = js_value["serialThreadCsvLogEnable"].get<bool>();
        }
        if (js_value.contains("serialThreadCsvLogPath") &&
            js_value["serialThreadCsvLogPath"].is_string()) {
            params.serialThreadCsvLogPath =
                js_value["serialThreadCsvLogPath"].get<std::string>();
        }
        if (js_value.contains("servoDebugLogEnable") &&
            js_value["servoDebugLogEnable"].is_boolean()) {
            params.servoDebugLogEnable = js_value["servoDebugLogEnable"].get<bool>();
        }
        if (js_value.contains("servoDebugLogPath") &&
            js_value["servoDebugLogPath"].is_string()) {
            params.servoDebugLogPath = js_value["servoDebugLogPath"].get<std::string>();
        }
        if (js_value.contains("imuAdDebugLogEnable") &&
            js_value["imuAdDebugLogEnable"].is_boolean()) {
            params.imuAdDebugLogEnable = js_value["imuAdDebugLogEnable"].get<bool>();
        }
        if (js_value.contains("imuAdDebugLogPath") &&
            js_value["imuAdDebugLogPath"].is_string()) {
            params.imuAdDebugLogPath = js_value["imuAdDebugLogPath"].get<std::string>();
        }
        if (js_value.contains("speedDecision") &&
            js_value["speedDecision"].is_object()) {
            const auto& spd = js_value["speedDecision"];
            if (spd.contains("enable") && spd["enable"].is_boolean()) {
                params.speedDecision.enable = spd["enable"].get<bool>();
            }
            if (spd.contains("obstacleVisibleSpeedMps") &&
                spd["obstacleVisibleSpeedMps"].is_number()) {
                params.speedDecision.obstacleVisibleSpeedMps =
                    spd["obstacleVisibleSpeedMps"].get<float>();
            }
            if (spd.contains("clearSpeedMps") &&
                spd["clearSpeedMps"].is_number()) {
                params.speedDecision.clearSpeedMps =
                    spd["clearSpeedMps"].get<float>();
            }
            if (spd.contains("virtualLookaheadSpeedMps") &&
                spd["virtualLookaheadSpeedMps"].is_number()) {
                params.speedDecision.virtualLookaheadSpeedMps =
                    spd["virtualLookaheadSpeedMps"].get<float>();
            }
            if (spd.contains("clearFrames") &&
                spd["clearFrames"].is_number_integer()) {
                params.speedDecision.clearFrames = spd["clearFrames"].get<int>();
            }
        }
        if (js_value.contains("virtualLookahead") &&
            js_value["virtualLookahead"].is_object()) {
            const auto& vl = js_value["virtualLookahead"];
            if (vl.contains("enable") && vl["enable"].is_boolean()) {
                params.virtualLookahead.enable = vl["enable"].get<bool>();
            }
            if (vl.contains("lowArrowMaxCount") &&
                vl["lowArrowMaxCount"].is_number_integer()) {
                params.virtualLookahead.lowArrowMaxCount =
                    vl["lowArrowMaxCount"].get<int>();
            }
            if (vl.contains("recoverArrowMinCount") &&
                vl["recoverArrowMinCount"].is_number_integer()) {
                params.virtualLookahead.recoverArrowMinCount =
                    vl["recoverArrowMinCount"].get<int>();
            }
            if (vl.contains("activateDistanceM") &&
                vl["activateDistanceM"].is_number()) {
                params.virtualLookahead.activateDistanceM =
                    vl["activateDistanceM"].get<float>();
            }
            if (vl.contains("fixedServoOutput") &&
                vl["fixedServoOutput"].is_number()) {
                params.virtualLookahead.fixedServoOutput =
                    vl["fixedServoOutput"].get<float>();
            }
        }
        if (js_value.contains("goldPosition") &&
            js_value["goldPosition"].is_object()) {
            const auto& gold_position = js_value["goldPosition"];
            if (gold_position.contains("correctionScale") &&
                gold_position["correctionScale"].is_number()) {
                params.goldPosition.correctionScale = std::clamp(
                    gold_position["correctionScale"].get<float>(),
                    0.0f,
                    1.0f);
            }
        }
        if (js_value.contains("goldPathPlanning") &&
            js_value["goldPathPlanning"].is_object()) {
            const auto& gold_path = js_value["goldPathPlanning"];
            if (gold_path.contains("goldBiasStepM") &&
                gold_path["goldBiasStepM"].is_number()) {
                params.goldPathPlanning.goldBiasStepM =
                    std::max(0.001f, gold_path["goldBiasStepM"].get<float>());
            }
            if (gold_path.contains("goldMaxBiasM") &&
                gold_path["goldMaxBiasM"].is_number()) {
                params.goldPathPlanning.goldMaxBiasM =
                    std::max(0.0f, gold_path["goldMaxBiasM"].get<float>());
            }
            if (gold_path.contains("goldCaptureCompensationM") &&
                gold_path["goldCaptureCompensationM"].is_number()) {
                params.goldPathPlanning.goldCaptureCompensationM =
                    std::max(0.0f,
                             gold_path["goldCaptureCompensationM"].get<float>());
            }
            if (gold_path.contains("goldCaptureToleranceM") &&
                gold_path["goldCaptureToleranceM"].is_number()) {
                params.goldPathPlanning.goldCaptureToleranceM =
                    std::max(0.0f,
                             gold_path["goldCaptureToleranceM"].get<float>());
            }
            if (gold_path.contains("goldBiasRateMps") &&
                gold_path["goldBiasRateMps"].is_number()) {
                params.goldPathPlanning.goldBiasRateMps =
                    std::max(0.0f, gold_path["goldBiasRateMps"].get<float>());
            }
            if (gold_path.contains("goldMaxNewDeltaM") &&
                gold_path["goldMaxNewDeltaM"].is_number()) {
                params.goldPathPlanning.goldMaxNewDeltaM =
                    std::max(0.0f,
                             gold_path["goldMaxNewDeltaM"].get<float>());
            }
            if (gold_path.contains("goldRecenterHoldFrames") &&
                gold_path["goldRecenterHoldFrames"].is_number_integer()) {
                params.goldPathPlanning.goldRecenterHoldFrames =
                    std::max(1,
                             gold_path["goldRecenterHoldFrames"].get<int>());
            }
            if (gold_path.contains("goldNearPriorityDistanceM") &&
                gold_path["goldNearPriorityDistanceM"].is_number()) {
                params.goldPathPlanning.goldNearPriorityDistanceM =
                    std::max(0.0f,
                             gold_path["goldNearPriorityDistanceM"].get<float>());
            }
            if (gold_path.contains("goldLateStartForwardM") &&
                gold_path["goldLateStartForwardM"].is_number()) {
                params.goldPathPlanning.goldLateStartForwardM =
                    std::max(0.0f,
                             gold_path["goldLateStartForwardM"].get<float>());
            }
            if (gold_path.contains("goldLateStartDeltaM") &&
                gold_path["goldLateStartDeltaM"].is_number()) {
                params.goldPathPlanning.goldLateStartDeltaM =
                    std::max(0.0f,
                             gold_path["goldLateStartDeltaM"].get<float>());
            }
            if (gold_path.contains("goldCenterReleaseM") &&
                gold_path["goldCenterReleaseM"].is_number()) {
                params.goldPathPlanning.goldCenterReleaseM =
                    std::max(0.0f, gold_path["goldCenterReleaseM"].get<float>());
            }
            if (gold_path.contains("pedestrianGoldDangerStripHalfWidthM") &&
                gold_path["pedestrianGoldDangerStripHalfWidthM"].is_number()) {
                params.goldPathPlanning.pedestrianGoldDangerStripHalfWidthM =
                    std::max(
                        0.0f,
                        gold_path["pedestrianGoldDangerStripHalfWidthM"].get<float>());
            }
            if (gold_path.contains("pedestrianGoldDangerStripHoldFrames") &&
                gold_path["pedestrianGoldDangerStripHoldFrames"].is_number_integer()) {
                params.goldPathPlanning.pedestrianGoldDangerStripHoldFrames =
                    std::max(
                        0,
                        gold_path["pedestrianGoldDangerStripHoldFrames"].get<int>());
            }
            if (gold_path.contains("terminalPoseEnable") &&
                gold_path["terminalPoseEnable"].is_boolean()) {
                params.goldPathPlanning.terminalPoseEnable =
                    gold_path["terminalPoseEnable"].get<bool>();
            }
            if (gold_path.contains("terminalPosePathToleranceM") &&
                gold_path["terminalPosePathToleranceM"].is_number()) {
                params.goldPathPlanning.terminalPosePathToleranceM =
                    std::max(0.0f,
                             gold_path["terminalPosePathToleranceM"].get<float>());
            }
            if (gold_path.contains("terminalPoseActivationDistanceM") &&
                gold_path["terminalPoseActivationDistanceM"].is_number()) {
                params.goldPathPlanning.terminalPoseActivationDistanceM =
                    std::max(0.0f,
                             gold_path["terminalPoseActivationDistanceM"].get<float>());
            }
            if (gold_path.contains("terminalPoseKp") &&
                gold_path["terminalPoseKp"].is_number()) {
                params.goldPathPlanning.terminalPoseKp =
                    std::max(0.0f, gold_path["terminalPoseKp"].get<float>());
            }
            if (gold_path.contains("terminalPoseKd") &&
                gold_path["terminalPoseKd"].is_number()) {
                params.goldPathPlanning.terminalPoseKd =
                    std::max(0.0f, gold_path["terminalPoseKd"].get<float>());
            }
            if (gold_path.contains("terminalPoseDerivativeFilterAlpha") &&
                gold_path["terminalPoseDerivativeFilterAlpha"].is_number()) {
                params.goldPathPlanning.terminalPoseDerivativeFilterAlpha =
                    std::clamp(
                        gold_path["terminalPoseDerivativeFilterAlpha"].get<float>(),
                        0.0f,
                        1.0f);
            }
            if (gold_path.contains("terminalPoseDerivativeLimitDeg") &&
                gold_path["terminalPoseDerivativeLimitDeg"].is_number()) {
                params.goldPathPlanning.terminalPoseDerivativeLimitDeg =
                    std::max(0.0f,
                             gold_path["terminalPoseDerivativeLimitDeg"].get<float>());
            }
            if (gold_path.contains("terminalPoseTargetJumpResetM") &&
                gold_path["terminalPoseTargetJumpResetM"].is_number()) {
                params.goldPathPlanning.terminalPoseTargetJumpResetM =
                    std::max(0.0f,
                             gold_path["terminalPoseTargetJumpResetM"].get<float>());
            }
            if (gold_path.contains("terminalPoseServoAbs") &&
                gold_path["terminalPoseServoAbs"].is_number()) {
                params.goldPathPlanning.terminalPoseServoAbs =
                    std::max(0.0f, gold_path["terminalPoseServoAbs"].get<float>());
            }
        }
        if (js_value.contains("speedLight") &&
            js_value["speedLight"].is_object()) {
            const auto& light = js_value["speedLight"];
            if (light.contains("stateMachineEnable") &&
                light["stateMachineEnable"].is_boolean()) {
                params.speedLight.stateMachineEnable =
                    light["stateMachineEnable"].get<bool>();
            }
        }
        if (js_value.contains("obstacleAvoidance") &&
            js_value["obstacleAvoidance"].is_object()) {
            const auto& obs = js_value["obstacleAvoidance"];
            if (obs.contains("enable") && obs["enable"].is_boolean()) {
                params.obstacleAvoidance.enable = obs["enable"].get<bool>();
            }
            if (obs.contains("confirmFrames") && obs["confirmFrames"].is_number_integer()) {
                params.obstacleAvoidance.confirmFrames = obs["confirmFrames"].get<int>();
            }
            if (obs.contains("exitLostFrames") && obs["exitLostFrames"].is_number_integer()) {
                params.obstacleAvoidance.exitLostFrames = obs["exitLostFrames"].get<int>();
            }
            if (obs.contains("prepareDistanceM") && obs["prepareDistanceM"].is_number()) {
                params.obstacleAvoidance.prepareDistanceM = obs["prepareDistanceM"].get<float>();
            }
            if (obs.contains("activateDistanceM") && obs["activateDistanceM"].is_number()) {
                params.obstacleAvoidance.activateDistanceM = obs["activateDistanceM"].get<float>();
            }
            if (obs.contains("vehicleLengthM") && obs["vehicleLengthM"].is_number()) {
                params.obstacleAvoidance.vehicleLengthM = obs["vehicleLengthM"].get<float>();
            }
            if (obs.contains("vehicleWidthM") && obs["vehicleWidthM"].is_number()) {
                params.obstacleAvoidance.vehicleWidthM = obs["vehicleWidthM"].get<float>();
            }
            if (obs.contains("vehicleFrontM") && obs["vehicleFrontM"].is_number()) {
                params.obstacleAvoidance.vehicleFrontM = obs["vehicleFrontM"].get<float>();
            }
            if (obs.contains("vehicleRearM") && obs["vehicleRearM"].is_number()) {
                params.obstacleAvoidance.vehicleRearM = obs["vehicleRearM"].get<float>();
            }
            if (obs.contains("forceLeftAvoidance") &&
                obs["forceLeftAvoidance"].is_boolean()) {
                params.obstacleAvoidance.forceLeftAvoidance =
                    obs["forceLeftAvoidance"].get<bool>();
            }
            if (obs.contains("vehicleAvoidanceRecoverHoldFrames") &&
                obs["vehicleAvoidanceRecoverHoldFrames"].is_number_integer()) {
                params.obstacleAvoidance.vehicleAvoidanceRecoverHoldFrames =
                    std::max(0,
                             obs["vehicleAvoidanceRecoverHoldFrames"].get<int>());
            }
            if (obs.contains("vehicleRecoverHoldMaxLifetimeFrames") &&
                obs["vehicleRecoverHoldMaxLifetimeFrames"].is_number_integer()) {
                params.obstacleAvoidance.vehicleRecoverHoldMaxLifetimeFrames =
                    std::max(
                        0,
                        obs["vehicleRecoverHoldMaxLifetimeFrames"].get<int>());
            }
            if (obs.contains("avoidOffsetM") && obs["avoidOffsetM"].is_number()) {
                params.obstacleAvoidance.avoidOffsetM = obs["avoidOffsetM"].get<float>();
            }
            if (obs.contains("vehicleAvoidanceEdgeClearanceM") &&
                obs["vehicleAvoidanceEdgeClearanceM"].is_number()) {
                params.obstacleAvoidance.vehicleAvoidanceEdgeClearanceM =
                    obs["vehicleAvoidanceEdgeClearanceM"].get<float>();
            }
            if (obs.contains("vehicleAvoidanceMaxOffsetM") &&
                obs["vehicleAvoidanceMaxOffsetM"].is_number()) {
                params.obstacleAvoidance.vehicleAvoidanceMaxOffsetM =
                    obs["vehicleAvoidanceMaxOffsetM"].get<float>();
            }
            if (obs.contains("vehicleCurveAvoidanceExtraOffsetM") &&
                obs["vehicleCurveAvoidanceExtraOffsetM"].is_number()) {
                params.obstacleAvoidance.vehicleCurveAvoidanceExtraOffsetM =
                    std::max(0.0f,
                             obs["vehicleCurveAvoidanceExtraOffsetM"].get<float>());
            }
            if (obs.contains("vehicleAvoidanceOppositeSteerThresholdDeg") &&
                obs["vehicleAvoidanceOppositeSteerThresholdDeg"].is_number()) {
                params.obstacleAvoidance.vehicleAvoidanceOppositeSteerThresholdDeg =
                    std::max(
                        0.0f,
                        obs["vehicleAvoidanceOppositeSteerThresholdDeg"].get<float>());
            }
            if (obs.contains("vehicleAvoidanceOppositeSteerHoldFrames") &&
                obs["vehicleAvoidanceOppositeSteerHoldFrames"].is_number_integer()) {
                params.obstacleAvoidance.vehicleAvoidanceOppositeSteerHoldFrames =
                    std::max(
                        0,
                        obs["vehicleAvoidanceOppositeSteerHoldFrames"].get<int>());
            }
            if (obs.contains("offsetRateMps") && obs["offsetRateMps"].is_number()) {
                params.obstacleAvoidance.offsetRateMps = obs["offsetRateMps"].get<float>();
            }
            if (obs.contains("offsetZeroEpsilonM") &&
                obs["offsetZeroEpsilonM"].is_number()) {
                params.obstacleAvoidance.offsetZeroEpsilonM =
                    obs["offsetZeroEpsilonM"].get<float>();
            }
            if (obs.contains("pedestrianDangerCorridorNearWidthM") &&
                obs["pedestrianDangerCorridorNearWidthM"].is_number()) {
                params.obstacleAvoidance.pedestrianDangerCorridorNearWidthM =
                    obs["pedestrianDangerCorridorNearWidthM"].get<float>();
            }
            if (obs.contains("pedestrianDangerCorridorFarWidthM") &&
                obs["pedestrianDangerCorridorFarWidthM"].is_number()) {
                params.obstacleAvoidance.pedestrianDangerCorridorFarWidthM =
                    obs["pedestrianDangerCorridorFarWidthM"].get<float>();
            }
            if (obs.contains("pedestrianDangerCorridorLengthM") &&
                obs["pedestrianDangerCorridorLengthM"].is_number()) {
                params.obstacleAvoidance.pedestrianDangerCorridorLengthM =
                    obs["pedestrianDangerCorridorLengthM"].get<float>();
            }
            if (obs.contains("pedestrianDangerCorridorSampleStepM") &&
                obs["pedestrianDangerCorridorSampleStepM"].is_number()) {
                params.obstacleAvoidance.pedestrianDangerCorridorSampleStepM =
                    obs["pedestrianDangerCorridorSampleStepM"].get<float>();
            }
            if (obs.contains("pedestrianServoStraightDeadband") &&
                obs["pedestrianServoStraightDeadband"].is_number()) {
                params.obstacleAvoidance.pedestrianServoStraightDeadband =
                    obs["pedestrianServoStraightDeadband"].get<float>();
            }
            if (obs.contains("pedestrianServoRadiusMaxAbs") &&
                obs["pedestrianServoRadiusMaxAbs"].is_number()) {
                params.obstacleAvoidance.pedestrianServoRadiusMaxAbs =
                    obs["pedestrianServoRadiusMaxAbs"].get<float>();
            }
            if (obs.contains("pedestrianDangerCorridorServoSmoothAlpha") &&
                obs["pedestrianDangerCorridorServoSmoothAlpha"].is_number()) {
                params.obstacleAvoidance.pedestrianDangerCorridorServoSmoothAlpha =
                    obs["pedestrianDangerCorridorServoSmoothAlpha"].get<float>();
            }
            if (obs.contains("pedestrianDangerCorridorServoMaxStep") &&
                obs["pedestrianDangerCorridorServoMaxStep"].is_number()) {
                params.obstacleAvoidance.pedestrianDangerCorridorServoMaxStep =
                    obs["pedestrianDangerCorridorServoMaxStep"].get<float>();
            }
            const bool pedestrian_corridor_geometry_valid =
                std::isfinite(params.obstacleAvoidance.pedestrianDangerCorridorNearWidthM) &&
                std::isfinite(params.obstacleAvoidance.pedestrianDangerCorridorFarWidthM) &&
                std::isfinite(params.obstacleAvoidance.pedestrianDangerCorridorLengthM) &&
                std::isfinite(params.obstacleAvoidance.pedestrianDangerCorridorSampleStepM) &&
                params.obstacleAvoidance.pedestrianDangerCorridorFarWidthM > 0.0f &&
                params.obstacleAvoidance.pedestrianDangerCorridorNearWidthM >=
                    params.obstacleAvoidance.pedestrianDangerCorridorFarWidthM &&
                params.obstacleAvoidance.pedestrianDangerCorridorLengthM > 0.0f &&
                params.obstacleAvoidance.pedestrianDangerCorridorSampleStepM > 0.0f;
            if (!pedestrian_corridor_geometry_valid) {
                const Params::ObstacleAvoidanceParams defaults;
                std::cerr
                    << "Invalid pedestrian curved-corridor dimensions; "
                    << "using defaults near/far/length/step="
                    << defaults.pedestrianDangerCorridorNearWidthM << "/"
                    << defaults.pedestrianDangerCorridorFarWidthM << "/"
                    << defaults.pedestrianDangerCorridorLengthM << "/"
                    << defaults.pedestrianDangerCorridorSampleStepM << " m\n";
                params.obstacleAvoidance.pedestrianDangerCorridorNearWidthM =
                    defaults.pedestrianDangerCorridorNearWidthM;
                params.obstacleAvoidance.pedestrianDangerCorridorFarWidthM =
                    defaults.pedestrianDangerCorridorFarWidthM;
                params.obstacleAvoidance.pedestrianDangerCorridorLengthM =
                    defaults.pedestrianDangerCorridorLengthM;
                params.obstacleAvoidance.pedestrianDangerCorridorSampleStepM =
                    defaults.pedestrianDangerCorridorSampleStepM;
            }
            params.obstacleAvoidance.pedestrianServoStraightDeadband =
                std::max(0.0f,
                         params.obstacleAvoidance.pedestrianServoStraightDeadband);
            params.obstacleAvoidance.pedestrianServoRadiusMaxAbs =
                std::max(params.obstacleAvoidance.pedestrianServoStraightDeadband,
                         params.obstacleAvoidance.pedestrianServoRadiusMaxAbs);
            params.obstacleAvoidance.pedestrianDangerCorridorServoSmoothAlpha =
                std::clamp(
                    params.obstacleAvoidance.pedestrianDangerCorridorServoSmoothAlpha,
                    0.0f,
                    1.0f);
            params.obstacleAvoidance.pedestrianDangerCorridorServoMaxStep =
                std::max(
                    0.0f,
                    params.obstacleAvoidance.pedestrianDangerCorridorServoMaxStep);
            if (obs.contains("pedestrianSlowEnable") &&
                obs["pedestrianSlowEnable"].is_boolean()) {
                params.obstacleAvoidance.pedestrianSlowEnable =
                    obs["pedestrianSlowEnable"].get<bool>();
            }
            if (obs.contains("pedestrianSlowDistanceM") &&
                obs["pedestrianSlowDistanceM"].is_number()) {
                params.obstacleAvoidance.pedestrianSlowDistanceM =
                    obs["pedestrianSlowDistanceM"].get<float>();
            }
            if (obs.contains("pedestrianSlowSpeedMps") &&
                obs["pedestrianSlowSpeedMps"].is_number()) {
                params.obstacleAvoidance.pedestrianSlowSpeedMps =
                    obs["pedestrianSlowSpeedMps"].get<float>();
            }
            if (obs.contains("pedestrianMotionWindowFrames") &&
                obs["pedestrianMotionWindowFrames"].is_number_integer()) {
                params.obstacleAvoidance.pedestrianMotionWindowFrames =
                    std::max(2, obs["pedestrianMotionWindowFrames"].get<int>());
            }
            if (obs.contains("pedestrianMotionTrendThresholdM") &&
                obs["pedestrianMotionTrendThresholdM"].is_number()) {
                params.obstacleAvoidance.pedestrianMotionTrendThresholdM =
                    std::max(0.0f,
                             obs["pedestrianMotionTrendThresholdM"].get<float>());
            }
            if (obs.contains("pedestrianMotionMaxJumpM") &&
                obs["pedestrianMotionMaxJumpM"].is_number()) {
                params.obstacleAvoidance.pedestrianMotionMaxJumpM =
                    std::max(0.0f, obs["pedestrianMotionMaxJumpM"].get<float>());
            }
            if (obs.contains("pedestrianSlowAwayConfirmFrames") &&
                obs["pedestrianSlowAwayConfirmFrames"].is_number_integer()) {
                params.obstacleAvoidance.pedestrianSlowAwayConfirmFrames =
                    std::max(1,
                             obs["pedestrianSlowAwayConfirmFrames"].get<int>());
            }
            if (obs.contains("pedestrianSlowObservationFrames") &&
                obs["pedestrianSlowObservationFrames"].is_number_integer()) {
                params.obstacleAvoidance.pedestrianSlowObservationFrames =
                    std::max(1,
                             obs["pedestrianSlowObservationFrames"].get<int>());
            }
            if (obs.contains("pedestrianSlowLostFrames") &&
                obs["pedestrianSlowLostFrames"].is_number_integer()) {
                params.obstacleAvoidance.pedestrianSlowLostFrames =
                    std::max(1, obs["pedestrianSlowLostFrames"].get<int>());
            }
            if (obs.contains("pedestrianStopClearFrames") &&
                obs["pedestrianStopClearFrames"].is_number_integer()) {
                params.obstacleAvoidance.pedestrianStopClearFrames =
                    std::max(1, obs["pedestrianStopClearFrames"].get<int>());
            }
            if (obs.contains("pedestrianReleaseLostGraceFrames") &&
                obs["pedestrianReleaseLostGraceFrames"].is_number_integer()) {
                params.obstacleAvoidance.pedestrianReleaseLostGraceFrames =
                    std::max(0,
                             obs["pedestrianReleaseLostGraceFrames"].get<int>());
            }
            if (obs.contains("pedestrianExitBoundaryMarginM") &&
                obs["pedestrianExitBoundaryMarginM"].is_number()) {
                params.obstacleAvoidance.pedestrianExitBoundaryMarginM =
                    std::max(0.0f,
                             obs["pedestrianExitBoundaryMarginM"].get<float>());
            }
            if (obs.contains("pedestrianStopCarSideThresholdM") &&
                obs["pedestrianStopCarSideThresholdM"].is_number()) {
                params.obstacleAvoidance.pedestrianStopCarSideThresholdM =
                    std::max(
                        0.0f,
                        obs["pedestrianStopCarSideThresholdM"].get<float>());
            }
            if (obs.contains("startupLaunchProtectFrames") &&
                obs["startupLaunchProtectFrames"].is_number_integer()) {
                params.obstacleAvoidance.startupLaunchProtectFrames =
                    std::max(1, obs["startupLaunchProtectFrames"].get<int>());
            }
            if (obs.contains("pedestrianStoppedLaunchSpeedMps") &&
                obs["pedestrianStoppedLaunchSpeedMps"].is_number()) {
                params.obstacleAvoidance.pedestrianStoppedLaunchSpeedMps =
                    std::max(
                        0.0f,
                        obs["pedestrianStoppedLaunchSpeedMps"].get<float>());
            }
            if (obs.contains("pedestrianSlowLaunchSpeedMps") &&
                obs["pedestrianSlowLaunchSpeedMps"].is_number()) {
                params.obstacleAvoidance.pedestrianSlowLaunchSpeedMps =
                    std::max(
                        0.0f,
                        obs["pedestrianSlowLaunchSpeedMps"].get<float>());
            }
            if (obs.contains("pathSmoothWindow") &&
                obs["pathSmoothWindow"].is_number_integer()) {
                params.obstacleAvoidance.pathSmoothWindow =
                    obs["pathSmoothWindow"].get<int>();
            }
            if (obs.contains("avoidLookaheadDistanceM") &&
                obs["avoidLookaheadDistanceM"].is_number()) {
                params.obstacleAvoidance.avoidLookaheadDistanceM =
                    obs["avoidLookaheadDistanceM"].get<float>();
            }
        }
    }
    catch (const nlohmann::detail::exception &e)
    {
        std::cerr << "Json Params Parse failed :" << e.what() << '\n';
        *flag = 0;
        return params;
    }

    *flag = 1;
    return params;
}
