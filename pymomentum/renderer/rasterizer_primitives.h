/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/camera/camera.h>
#include <momentum/rasterizer/fwd.h>
#include <momentum/rasterizer/rasterizer.h>
#include <momentum/rasterizer/text_rasterizer.h>

#include <pybind11/numpy.h>
#include <Eigen/Geometry>
#include <cfloat>

#include <optional>

namespace pymomentum {

pybind11::array_t<float> createZBuffer(const momentum::Camera& camera, float far_clip = FLT_MAX);
pybind11::array_t<float> createRGBBuffer(
    const momentum::Camera& camera,
    std::optional<Eigen::Vector3f> bgColor);
pybind11::array_t<int32_t> createIndexBuffer(const momentum::Camera& camera);

momentum::rasterizer::PhongMaterial createPhongMaterial(
    const std::optional<Eigen::Vector3f>& diffuseColor,
    const std::optional<Eigen::Vector3f>& specularColor,
    const std::optional<float>& specularExponent,
    const std::optional<Eigen::Vector3f>& emissiveColor,
    const std::optional<pybind11::array_t<const float>>& diffuseTexture,
    const std::optional<pybind11::array_t<const float>>& emissiveTexture);

void alphaMatte(
    const pybind11::buffer& zBuffer,
    const pybind11::buffer& rgbBuffer,
    const pybind11::array& tgtRgbImage,
    float alpha = 1.0f);

void rasterizeLines(
    pybind11::buffer positions,
    const momentum::Camera& camera,
    const pybind11::buffer& zBuffer,
    const std::optional<pybind11::buffer>& rgbBuffer,
    float width,
    const std::optional<Eigen::Vector3f>& color,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset);

void rasterizeCircles(
    pybind11::buffer positions,
    const momentum::Camera& camera,
    const pybind11::buffer& zBuffer,
    const std::optional<pybind11::buffer>& rgbBuffer,
    float lineThickness,
    float radius,
    const std::optional<Eigen::Vector3f>& lineColor,
    std::optional<Eigen::Vector3f> fillColor,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset);

void rasterizeCameraFrustum(
    const momentum::Camera& frustumCamera,
    const momentum::Camera& camera,
    const pybind11::buffer& zBuffer,
    const std::optional<pybind11::buffer>& rgbBuffer,
    float lineWidth,
    float distance,
    size_t numSamples,
    const std::optional<Eigen::Vector3f>& color,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset);

/// Rasterize a grid of lines in the x-z plane (with y up).
void rasterizeGrid(
    const momentum::Camera& camera,
    const pybind11::buffer& zBuffer,
    const std::optional<pybind11::buffer>& rgbBuffer,
    float thickness,
    const std::optional<Eigen::Vector3f>& color,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset,
    float width,
    int numLines);

// 2D rasterization functions that operate directly in image space
// without camera projection or z-buffer

void rasterizeLines2D(
    pybind11::buffer positions,
    const pybind11::buffer& rgbBuffer,
    float thickness,
    const std::optional<Eigen::Vector3f>& color,
    const std::optional<pybind11::buffer>& zBuffer,
    const std::optional<Eigen::Vector2f>& imageOffset);

void rasterizeText(
    pybind11::buffer positions,
    const std::vector<std::string>& texts,
    const momentum::Camera& camera,
    const pybind11::buffer& zBuffer,
    const std::optional<pybind11::buffer>& rgbBuffer,
    const std::optional<Eigen::Vector3f>& color,
    int textScale,
    momentum::rasterizer::HorizontalAlignment horizontalAlignment,
    momentum::rasterizer::VerticalAlignment verticalAlignment,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset);

void rasterizeText2D(
    pybind11::buffer positions,
    const std::vector<std::string>& texts,
    const pybind11::buffer& rgbBuffer,
    const std::optional<Eigen::Vector3f>& color,
    int textScale,
    momentum::rasterizer::HorizontalAlignment horizontalAlignment,
    momentum::rasterizer::VerticalAlignment verticalAlignment,
    const std::optional<pybind11::buffer>& zBuffer,
    const std::optional<Eigen::Vector2f>& imageOffset);

void rasterizeCircles2D(
    pybind11::buffer positions,
    const pybind11::buffer& rgbBuffer,
    float lineThickness,
    float radius,
    const std::optional<Eigen::Vector3f>& lineColor,
    std::optional<Eigen::Vector3f> fillColor,
    const std::optional<pybind11::buffer>& zBuffer,
    const std::optional<Eigen::Vector2f>& imageOffset);

void rasterizeTransforms(
    pybind11::buffer transforms,
    const momentum::Camera& camera,
    const pybind11::buffer& zBuffer,
    const std::optional<pybind11::buffer>& rgbBuffer,
    const std::optional<pybind11::buffer>& surfaceNormalsBuffer,
    float scale,
    const std::optional<momentum::rasterizer::PhongMaterial>& material,
    std::optional<std::vector<momentum::rasterizer::Light>> lights,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset,
    int lengthSubdivisions,
    int radiusSubdivisions);

} // namespace pymomentum
