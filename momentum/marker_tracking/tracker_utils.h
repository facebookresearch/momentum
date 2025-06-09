/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/character/locator.h>
#include <momentum/character/marker.h>
#include <momentum/character_solver/fwd.h>

namespace momentum {

std::vector<std::vector<momentum::PositionData>> createConstraintData(
    gsl::span<const std::vector<momentum::Marker>> markerData,
    const momentum::LocatorList& locators);

// TODO: remove the one in momentum

// Create a LocatorCharacter where each locator is a bone in its skeleton. This character is used
// for calibrating locator offsets (as bone offset parameters).
momentum::Character createLocatorCharacter(
    const momentum::Character& sourceCharacter,
    const std::string& prefix);

// Extract locator offsets from a LocatorCharacter for a normal Character given input calibrated
// parameters
momentum::LocatorList extractLocatorsFromCharacter(
    const momentum::Character& locatorCharacter,
    const momentum::CharacterParameters& calibParams);

// TODO: move to momentum proper
momentum::ModelParameters extractParameters(
    const momentum::ModelParameters& params,
    const momentum::ParameterSet& parameterSet);

std::tuple<Eigen::VectorXf, momentum::LocatorList> extractIdAndLocatorsFromParams(
    const momentum::ModelParameters& param,
    const momentum::Character& sourceCharacter,
    const momentum::Character& targetCharacter);

void fillIdentity(
    const momentum::ParameterSet& idSet,
    const momentum::ModelParameters& identity,
    Eigen::MatrixXf& motion);

void removeIdentity(
    const momentum::ParameterSet& idSet,
    const momentum::ModelParameters& identity,
    Eigen::MatrixXf& motion);

std::vector<std::vector<momentum::Marker>> extractMarkersFromMotion(
    const momentum::Character& character,
    const Eigen::MatrixXf& motion);

} // namespace momentum
