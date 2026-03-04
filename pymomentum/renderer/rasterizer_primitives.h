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

#include <ATen/ATen.h>
#include <pybind11/numpy.h>
#include <Eigen/Geometry>
#include <cfloat>

#include <optional>

namespace pymomentum {

at::Tensor createZBuffer(const momentum::Camera& camera, float far_clip = FLT_MAX);
at::Tensor createRGBBuffer(const momentum::Camera& camera, std::optional<Eigen::Vector3f> bgColor);
at::Tensor createIndexBuffer(const momentum::Camera& camera);

momentum::rasterizer::PhongMaterial createPhongMaterial(
    const std::optional<Eigen::Vector3f>& diffuseColor,
    const std::optional<Eigen::Vector3f>& specularColor,
    const std::optional<float>& specularExponent,
    const std::optional<Eigen::Vector3f>& emissiveColor,
    const std::optional<pybind11::array_t<const float>>& diffuseTexture,
    const std::optional<pybind11::array_t<const float>>& emissiveTexture);

void alphaMatte(
    at::Tensor zBuffer,
    at::Tensor rgbBuffer,
    const pybind11::array& tgtRgbImage,
    float alpha = 1.0f);

void rasterizeLines(
    at::Tensor positions,
    const momentum::Camera& camera,
    at::Tensor zBuffer,
    std::optional<at::Tensor> rgbBuffer,
    float width,
    const std::optional<Eigen::Vector3f>& color,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset);

void rasterizeCircles(
    at::Tensor positions,
    const momentum::Camera& camera,
    at::Tensor zBuffer,
    std::optional<at::Tensor> rgbBuffer,
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
    at::Tensor zBuffer,
    std::optional<at::Tensor> rgbBuffer,
    float lineWidth,
    float distance,
    size_t numSamples,
    const std::optional<Eigen::Vector3f>& color,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset);

// 2D rasterization functions that operate directly in image space
// without camera projection or z-buffer

void rasterizeLines2D(
    at::Tensor positions,
    at::Tensor rgbBuffer,
    float thickness,
    const std::optional<Eigen::Vector3f>& color,
    std::optional<at::Tensor> zBuffer,
    const std::optional<Eigen::Vector2f>& imageOffset);

void rasterizeText(
    at::Tensor positions,
    const std::vector<std::string>& texts,
    const momentum::Camera& camera,
    at::Tensor zBuffer,
    std::optional<at::Tensor> rgbBuffer,
    const std::optional<Eigen::Vector3f>& color,
    int textScale,
    momentum::rasterizer::HorizontalAlignment horizontalAlignment,
    momentum::rasterizer::VerticalAlignment verticalAlignment,
    const std::optional<Eigen::Matrix4f>& modelMatrix,
    float nearClip,
    float depthOffset,
    const std::optional<Eigen::Vector2f>& imageOffset);

void rasterizeText2D(
    at::Tensor positions,
    const std::vector<std::string>& texts,
    at::Tensor rgbBuffer,
    const std::optional<Eigen::Vector3f>& color,
    int textScale,
    momentum::rasterizer::HorizontalAlignment horizontalAlignment,
    momentum::rasterizer::VerticalAlignment verticalAlignment,
    std::optional<at::Tensor> zBuffer,
    const std::optional<Eigen::Vector2f>& imageOffset);

void rasterizeCircles2D(
    at::Tensor positions,
    at::Tensor rgbBuffer,
    float lineThickness,
    float radius,
    const std::optional<Eigen::Vector3f>& lineColor,
    std::optional<Eigen::Vector3f> fillColor,
    std::optional<at::Tensor> zBuffer,
    const std::optional<Eigen::Vector2f>& imageOffset);

void rasterizeTransforms(
    at::Tensor transforms,
    const momentum::Camera& camera,
    at::Tensor zBuffer,
    std::optional<at::Tensor> rgbBuffer,
    std::optional<at::Tensor> surfaceNormalsBuffer,
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
