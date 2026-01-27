/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/geometry/character_pybind.h"

#include "pymomentum/geometry/array_skinning.h"
#include "pymomentum/geometry/momentum_geometry.h"
#include "pymomentum/geometry/momentum_io.h"
#include "pymomentum/tensor_momentum/tensor_parameter_transform.h"
#include "pymomentum/tensor_momentum/tensor_skinning.h"
#include "pymomentum/torch_bridge.h"

#include <momentum/character/blend_shape.h>
#include <momentum/character/character.h>
#include <momentum/character/character_utility.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/io/fbx/fbx_io.h>
#include <momentum/io/gltf/gltf_io.h>
#include <momentum/io/legacy_json/legacy_json_io.h>
#include <momentum/io/marker/coordinate_system.h>
#include <momentum/math/mesh.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Core>

#include <algorithm>
#include <limits>
#include <memory>

#include <fmt/format.h>

namespace py = pybind11;
namespace mm = momentum;

namespace {

mm::Character withBlendShapeImpl(
    const mm::Character& c,
    const std::optional<mm::BlendShape_const_p>& blendShape,
    int nShapes) {
  if (!blendShape.has_value()) {
    // Remove existing blend shape, if present:
    return c.withBlendShape({}, 0);
  }

  MT_THROW_IF(!c.mesh, "Character has no mesh, cannot apply blend shapes.")

  auto blendShapePtr = blendShape.value();
  MT_THROW_IF(
      blendShapePtr->modelSize() != c.mesh->vertices.size(),
      "Blend shape has {} vertices, but mesh has {} vertices.",
      blendShapePtr->modelSize(),
      c.mesh->vertices.size());
  return c.withBlendShape(blendShapePtr, nShapes < 0 ? INT_MAX : nShapes);
}

} // namespace

namespace pymomentum {

void registerCharacterBindings(py::class_<mm::Character>& characterClass) {
  // =====================================================
  // momentum::Character
  // - name
  // - metadata
  // - skeleton
  // - parameter_transform
  // - locators
  // - mesh
  // - skin_weights
  // - blend_shape
  // - collision_geometry
  // - model_parameter_limits
  // - joint_parameter_limits
  // - [constructor](name, skeleton, parameter_transform, locators)
  // - with_mesh_and_skin_weights(mesh, skin_weights)
  // - with_blend_shape(blend_shape, n_shapes)
  // - with_face_expression_blend_shape(blend_shape, n_shapes)
  //
  // [member methods]
  // - pose_mesh(jointParams)
  // - skin_points(skel_state, rest_vertices)
  // - scaled(scale)
  // - transformed(xform)
  // - rebind_skin()
  // - find_locators(names)
  // - apply_model_param_limits(model_params)
  // - simplify(enabled_parameters)
  // - load_locators(filename)
  // - load_locators_from_bytes(locator_bytes)
  // - load_model_definition(filename)
  // - load_model_definition_from_bytes(model_bytes)
  //
  // [static methods for io]
  // - load_gltf_from_bytes(gltf_btyes)
  // - to_gltf(character, fps, motion, offsets)
  // - load_fbx(fbxFilename, modelFilename, locatorsFilename)
  // - load_fbx_from_bytes(fbx_bytes, permissive)
  // - load_fbx_with_motion(fbxFilename, permissive, strip_namespaces)
  // - load_fbx_with_motion_from_bytes(fbx_bytes, permissive, strip_namespaces)
  // - load_gltf(path)
  // - load_gltf_with_motion(gltfFilename)
  // - load_urdf(urdf_filename)
  // - load_urdf_from_bytes(urdf_bytes)
  // - save_gltf(path, character, fps, motion, offsets, markers, options)
  // - save_gltf_from_skel_states(path, character, fps, skel_states,
  // joint_params, markers, options)
  // - save_fbx(path, character, fps, motion, offsets, coord_system_info, markers, fbx_namespace)
  // - save_fbx_with_joint_params(path, character, fps, joint_params, coord_system_info,
  // markers, fbx_namespace)
  // =====================================================
  characterClass
      .def(
          py::init([](const std::string& name,
                      const mm::Skeleton& skeleton,
                      const mm::ParameterTransform& parameterTransform,
                      const mm::LocatorList& locators = mm::LocatorList()) {
            auto character = mm::Character(skeleton, parameterTransform);
            character.name = name;
            character.locators = locators;
            return character;
          }),
          py::arg("name"),
          py::arg("skeleton"),
          py::arg("parameter_transform"),
          py::kw_only(),
          py::arg("locators") = mm::LocatorList())
      .def(
          "with_mesh_and_skin_weights",
          [](const mm::Character& character,
             const mm::Mesh& mesh,
             const std::optional<mm::SkinWeights>& skinWeights) {
            if (skinWeights) {
              MT_THROW_IF(
                  skinWeights->index.rows() != skinWeights->weight.rows(),
                  "The number of rows in the index and weight matrices should match; got {} and {}.",
                  skinWeights->index.rows(),
                  skinWeights->weight.rows());

              MT_THROW_IF(
                  skinWeights->index.maxCoeff() >= character.skeleton.joints.size(),
                  "Skin weight index is out of range; max index is {}, but there are only {} joints.",
                  skinWeights->index.maxCoeff(),
                  character.skeleton.joints.size());
            }

            const mm::SkinWeights* skinWeightsPtr = character.skinWeights.get();
            if (skinWeights) {
              skinWeightsPtr = &skinWeights.value();
            }

            if (skinWeightsPtr) {
              MT_THROW_IF(
                  skinWeightsPtr->weight.rows() != mesh.vertices.size(),
                  "The number of mesh vertices and skin weight index/weight matrix rows should be the same {} vs {}",
                  mesh.vertices.size(),
                  skinWeightsPtr->index.rows());
            }

            return momentum::Character(
                character.skeleton,
                character.parameterTransform,
                character.parameterLimits,
                character.locators,
                &mesh,
                skinWeightsPtr,
                character.collision.get(),
                character.poseShapes.get(),
                character.blendShape,
                character.faceExpressionBlendShape,
                character.name,
                character.inverseBindPose,
                character.skinnedLocators,
                character.metadata);
          },
          "Adds mesh and skin weight to the character and return a new character instance",
          py::arg("mesh"),
          py::arg("skin_weights") = std::optional<mm::SkinWeights>{})
      .def(
          "with_parameter_limits",
          [](const mm::Character& character,
             const std::vector<mm::ParameterLimit>& parameterLimits) {
            return mm::Character(
                character.skeleton,
                character.parameterTransform,
                parameterLimits,
                character.locators,
                character.mesh.get(),
                character.skinWeights.get(),
                character.collision.get(),
                character.poseShapes.get(),
                character.blendShape,
                character.faceExpressionBlendShape,
                character.name,
                character.inverseBindPose,
                character.skinnedLocators,
                character.metadata);
          },
          "Returns a new character with the parameter limits set to the passed-in limits.",
          py::arg("parameter_limits"))
      .def(
          "clone",
          [](const mm::Character& character) { return mm::Character{character}; },
          "Performs a deep-copy of the character.")
      .def(
          "with_locators",
          [](const mm::Character& character,
             const momentum::LocatorList& locators,
             bool replace = false) {
            momentum::LocatorList combinedLocators;
            if (!replace) {
              std::copy(
                  character.locators.begin(),
                  character.locators.end(),
                  std::back_inserter(combinedLocators));
            }
            std::copy(locators.begin(), locators.end(), std::back_inserter(combinedLocators));
            return momentum::Character(
                character.skeleton,
                character.parameterTransform,
                character.parameterLimits,
                combinedLocators,
                character.mesh.get(),
                character.skinWeights.get(),
                character.collision.get(),
                character.poseShapes.get(),
                character.blendShape,
                character.faceExpressionBlendShape,
                character.name,
                character.inverseBindPose,
                character.skinnedLocators,
                character.metadata);
          },
          R"(Returns a new character with the passed-in locators.  If 'replace' is true, the existing locators are replaced, otherwise (the default) the new locators are appended to the existing ones.

          :param locators: The locators to add to the character.
          :param replace: If true, replace the existing locators with the passed-in ones.  Otherwise, append the new locators to the existing ones.  Defaults to false.
          )",
          py::arg("locators"),
          py::arg("replace") = false)
      .def(
          "with_skinned_locators",
          [](const mm::Character& character,
             const momentum::SkinnedLocatorList& skinnedLocators,
             bool replace = false) {
            for (const auto& skinnedLocator : skinnedLocators) {
              for (Eigen::Index i = 0; i < skinnedLocator.parents.size(); ++i) {
                if (skinnedLocator.parents[i] >= character.skeleton.joints.size()) {
                  throw py::index_error(
                      fmt::format(
                          "Skinned locator {} has parent index {} which is out of range (there are only {} joints).",
                          skinnedLocator.name,
                          skinnedLocator.parents[i],
                          character.skeleton.joints.size()));
                }
              }
            }

            momentum::SkinnedLocatorList combinedSkinnedLocators;
            if (!replace) {
              std::copy(
                  character.skinnedLocators.begin(),
                  character.skinnedLocators.end(),
                  std::back_inserter(combinedSkinnedLocators));
            }
            std::copy(
                skinnedLocators.begin(),
                skinnedLocators.end(),
                std::back_inserter(combinedSkinnedLocators));
            return momentum::Character(
                character.skeleton,
                character.parameterTransform,
                character.parameterLimits,
                character.locators,
                character.mesh.get(),
                character.skinWeights.get(),
                character.collision.get(),
                character.poseShapes.get(),
                character.blendShape,
                character.faceExpressionBlendShape,
                character.name,
                character.inverseBindPose,
                combinedSkinnedLocators,
                character.metadata);
          },
          R"(Returns a new character with the passed-in skinned locators.  If 'replace' is true, the existing skinned locators are replaced, otherwise (the default) the new skinned locators are appended to the existing ones.

          :param skinned_locators: The skinned locators to add to the character.
          :param replace: If true, replace the existing skinned locators with the passed-in ones.  Otherwise, append the new skinned locators to the existing ones.  Defaults to false.
          )",
          py::arg("skinned_locators"),
          py::arg("replace") = false)
      .def_readonly("name", &mm::Character::name, "The character's name.")
      .def_readonly("metadata", &mm::Character::metadata, "The character's metadata.")
      .def_readonly(
          "skeleton", &mm::Character::skeleton, "The character's skeleton. See :class:`Skeleton`.")
      .def_readonly(
          "parameter_limits",
          &mm::Character::parameterLimits,
          "The character's parameter limits. See :class:`ParameterLimit`.")
      .def_readonly(
          "parameter_transform",
          &mm::Character::parameterTransform,
          "Maps the reduced k-dimensional modelParameters that are used in the IK solve "
          "to the full 7*n-dimensional parameters used in the skeleton. See :class:`ParameterTransform`.")
      .def_readonly(
          "locators",
          &mm::Character::locators,
          "List of locators on the mesh. See :class:`Locator`.")
      .def_readonly(
          "skinned_locators",
          &mm::Character::skinnedLocators,
          "List of skinned locators on the mesh.")
      .def_property_readonly(
          "mesh",
          [](const mm::Character& c) -> std::unique_ptr<mm::Mesh> {
            return (c.mesh) ? std::make_unique<mm::Mesh>(*c.mesh) : mm::Mesh_u();
          },
          ":return: The character's :class:`Mesh`, or None if not present.")
      .def_property_readonly(
          "has_mesh",
          [](const mm::Character& c) -> bool {
            return static_cast<bool>(c.mesh) && static_cast<bool>(c.skinWeights);
          })
      .def_property_readonly(
          "skin_weights",
          [](const mm::Character& c) -> std::unique_ptr<mm::SkinWeights> {
            return (c.skinWeights) ? std::make_unique<mm::SkinWeights>(*c.skinWeights)
                                   : mm::SkinWeights_u();
          },
          "The character's skinning weights. See :class:`SkinWeights`.")
      .def_property_readonly(
          "blend_shape",
          [](const mm::Character& c) -> std::optional<std::shared_ptr<const mm::BlendShape>> {
            if (c.blendShape) {
              return c.blendShape;
            } else {
              return {};
            }
          },
          ":return: The character's :class:`BlendShape` basis, if present, or None.")
      .def_property_readonly(
          "face_expression_blend_shape",
          [](const mm::Character& c) -> std::optional<std::shared_ptr<const mm::BlendShapeBase>> {
            if (c.faceExpressionBlendShape) {
              return c.faceExpressionBlendShape;
            } else {
              return {};
            }
          },
          ":return: The character's :class:`BlendShapeBase` basis, if present, or None.")
      .def_property_readonly(
          "collision_geometry",
          [](const mm::Character& c) -> mm::CollisionGeometry {
            if (c.collision) {
              return *c.collision;
            } else {
              return {};
            }
          },
          ":return: A list of :class:`TaperedCapsule` representing the character's collision geometry.")
      .def(
          "with_blend_shape",
          &withBlendShapeImpl,
          R"(Returns a character that uses the parameter transform to control the passed-in blend shape basis.
It can be used to solve for shapes and pose simultaneously.

:param blend_shape: Blend shape basis.
:param n_shapes: Max blend shapes to retain.  Pass -1 to keep all of them (but warning: the default allgender basis is quite large with hundreds of shapes).
)",
          py::arg("blend_shape"),
          py::arg("n_shapes") = -1)
      .def(
          "with_face_expression_blend_shape",
          [](const mm::Character& c,
             const std::optional<mm::BlendShapeBase_const_p>& blendShape,
             int nShapes) {
            return c.withFaceExpressionBlendShape(
                blendShape.value_or(mm::BlendShapeBase_const_p{}), nShapes < 0 ? INT_MAX : nShapes);
          },
          R"(Returns a character that uses the parameter transform to control the passed-in blend shapes.
It can be used to solve for facial expressions.

:param blend_shape: Blend shape basis (shape vectors only).
:param n_shapes: Max blend shapes to retain.  Pass -1 to keep all of them.
)",
          py::arg("blend_shape"),
          py::arg("n_shapes") = -1)
      .def(
          "with_collision_geometry",
          [](const mm::Character& c, const std::vector<mm::TaperedCapsule>& collision_geometry) {
            return mm::Character(
                c.skeleton,
                c.parameterTransform,
                c.parameterLimits,
                c.locators,
                c.mesh.get(),
                c.skinWeights.get(),
                &collision_geometry,
                c.poseShapes.get(),
                c.blendShape,
                c.faceExpressionBlendShape,
                c.name,
                c.inverseBindPose,
                c.skinnedLocators,
                c.metadata);
          },
          "Returns a new :class:`Character` with the collision geometry replaced.")
      .def(
          "bake_blend_shape",
          [](const mm::Character& c, const py::array_t<float>& blendWeights) {
            // Convert array to BlendWeights
            MT_THROW_IF(
                blendWeights.ndim() != 1,
                "blend_weights must be a 1D array, got {}D array",
                blendWeights.ndim());

            auto unchecked = blendWeights.unchecked<1>();

            // Create BlendWeights from the array data
            mm::BlendWeights weights(blendWeights.shape(0));
            for (int k = 0; k < blendWeights.shape(0); ++k) {
              weights.v(k) = unchecked(k);
            }

            return c.bakeBlendShape(weights);
          },
          R"(Returns a new :class:`Character` with blend shapes baked into the mesh.

:param blend_weights: A 1D array of blend shape weights to apply.
:return: A new :class:`Character` with the blend shapes baked into the mesh and blend shape parameters removed from the parameter transform.)",
          py::arg("blend_weights"))
      .def(
          "pose_mesh",
          &pymomentum::getPosedMesh,
          R"(Poses the mesh

:param joint_params: The (7*nJoints) joint parameters for the given pose.
:return: A :class:`Mesh` object with the given pose.)",
          py::arg("joint_params"))
      .def(
          "skin_points",
          &pymomentum::skinPointsArray,
          R"(Skins the points using the character's linear blend skinning.

:param skel_state: An ndarray containing either:

    - Skeleton state with shape [..., nJoints, 8] where each joint has [tx, ty, tz, rx, ry, rz, rw, scale], OR
    - Transform matrices with shape [..., nJoints, 4, 4] containing 4x4 transformation matrices.

:param rest_vertices: An optional ndarray containing the rest points with shape [..., nVertices, 3]. If not passed, the ones stored inside the character are used.
:return: The vertex positions in worldspace with shape [..., nVertices, 3].
          )",
          py::arg("skel_state"),
          py::arg("rest_vertices") = std::optional<py::buffer>{})
      .def(
          "skin_skinned_locators",
          [](const momentum::Character& character,
             const at::Tensor& skel_state,
             const std::optional<at::Tensor>& rest_positions) {
            return skinSkinnedLocators(character, skel_state, rest_positions);
          },
          R"(Apply linear blend skinning to compute the world-space positions of the character's skinned locators.

This function uses the character's built-in skinned locators and applies linear blend skinning
to compute their world-space positions given a skeleton state.

:param skel_state: Skeleton state tensor with shape [nJoints x 8] or [nBatch x nJoints x 8].
:param rest_positions: Optional rest positions tensor with shape [nLocators x 3] or [nBatch x nLocators x 3]. If not provided, uses the position stored in each SkinnedLocator.
:return: Tensor of shape [nLocators x 3] or [nBatch x nLocators x 3] containing the world-space positions of the skinned locators.
)",
          py::arg("skel_state"),
          py::arg("rest_positions") = std::optional<at::Tensor>())
      .def(
          "scaled",
          &momentum::scaleCharacter,
          R"(Scale the character (mesh and skeleton) by the desired amount.

Note that this primarily be used when transforming the character into different units; if you
simply want to apply an identity-specific scale to the character, you should use the
'scale_global' parameter in the :class:`ParameterTransform`.

:return: a new :class:`Character` that has been scaled.
:param character: The character to be scaled.
:param scale: The scale to apply.)",
          py::arg("scale"))
      .def(
          "transformed",
          [](const momentum::Character& character, const Eigen::Matrix4f& xform) {
            return momentum::transformCharacter(character, Eigen::Affine3f(xform));
          },
          R"(Transform the character (mesh and skeleton) by the desired transformation matrix.

Note that this is primarily intended for transforming between different spaces (e.g. x-up vs y-up).
If you want to translate/rotate/scale a character, you should preferentially use the model parameters to do so.

:return: a new :class:`Character` that has been transformed.
:param character: The character to be transformed.
:param xform: The transform to apply.)",
          py::arg("xform"))
      .def(
          "rebind_skin",
          [](const momentum::Character& character) {
            momentum::Character result(character);
            result.initInverseBindPose();
            return result;
          },
          "Rebind the character's inverse bind pose from the resting skeleton pose.")
      .def_property_readonly("bind_pose", &getBindPose, "Get the bind pose for skinning.")
      .def_property_readonly(
          "inverse_bind_pose", &getInverseBindPose, "Get the inverse bind pose for skinning.")
      .def(
          "find_locators",
          &getLocators,
          R"(Return the parents/offsets of the passed-in locators.

:param names: The names of the locators or joints.
:return: a pair [parents, offsets] of numpy arrays.)",
          py::arg("names"))
      .def(
          "apply_model_param_limits",
          &applyModelParameterLimits,
          R"(Clamp model parameters by parameter limits stored in Character.

Note the function is differentiable.

:param model_params: the (can be batched) body model parameters.
:return: clampled model parameters. Same tensor shape as the input.)",
          py::arg("model_params"))
      .def_property_readonly(
          "model_parameter_limits",
          &modelParameterLimits,
          R"(A tuple (min, max) where each is an nParameter-length ndarray containing the upper or lower limits for the model parameters.  Note that not all parameters will have limits; for those parameters (such as global translation) without limits, (-FLT_MAX, FLT_MAX) is returned.)")
      .def_property_readonly(
          "joint_parameter_limits",
          &jointParameterLimits,
          R"(A tuple (min, max) where each is an (nJoints x 7)-length ndarray containing the upper or lower limits for the joint parameters.

Note that not all parameters will have limits; for those parameters (such as global translation) without limits, (-FLT_MAX, FLT_MAX) is returned.

Note: In practice, most limits are enforced on the model parameters, but momentum's joint limit functionality permits applying limits to joint parameters also as a conveninence.  )")
      .def_static(
          "load_gltf_from_bytes",
          &loadGLTFCharacterFromBytes,
          R"(Load a character from a gltf byte array.

:param gltf_bytes: A :class:`bytes` containing the GLTF JSON/messagepack data.
:return: a valid :class:`Character`.
      )",
          py::arg("gltf_bytes"))
      .def_static(
          "load_gltf_with_motion_from_bytes",
          &loadGLTFCharacterWithMotionFromBytes,
          R"(Load a character and motion from a gltf byte array.

  :param gltf_bytes: A :class:`bytes` containing the GLTF JSON/messagepack data.
  :return: a valid :class:`Character`.
        )",
          py::arg("gltf_bytes"))
      .def_static(
          "load_gltf_with_motion_model_parameter_scales",
          &loadGLTFCharacterWithMotionModelParameterScales,
          py::call_guard<py::gil_scoped_release>(),
          R"(Load BOTH character (skeleton/mesh) and motion data from a gltf file in a single call,
with model parameter scales instead of joint parameters.

This function differs from :meth:`load_gltf_with_motion` by returning identity parameters
as model parameters (numAllModelParameters,) instead of joint parameters
(numJoints * kParametersPerJoint,). Additionally, the motion has the identity parameters
added back to the scale parameters.

Use this function when you need the character structure, animation, and identity parameters
all expressed in model parameter space, which is the natural format for optimization and
character posing.

:param gltf_filename: Path to the GLTF/GLB file to load.
:return: A tuple (character, motion, model_identity, fps), where character is the loaded :class:`Character` object,
         motion is a numpy array of shape (numFrames, numAllModelParameters) with identity added to scale parameters,
         model_identity is the identity/scale parameters as a numpy array of shape (numAllModelParameters,),
         and fps is the frames per second of the motion data.
)",
          py::arg("gltf_filename"))
      .def_static(
          "load_gltf_with_motion_model_parameter_scales_from_bytes",
          &loadGLTFCharacterWithMotionModelParameterScalesFromBytes,
          R"(Load BOTH character (skeleton/mesh) and motion data from a gltf byte array,
with model parameter scales instead of joint parameters.

This is the byte array version of :meth:`load_gltf_with_motion_model_parameter_scales`.
The function returns identity parameters as model parameters and adds them to the
motion's scale parameters.

:param gltf_bytes: A :class:`bytes` containing the GLTF JSON/messagepack data.
:return: A tuple (character, motion, model_identity, fps), where character is the loaded :class:`Character` object,
         motion is a numpy array of shape (numFrames, numAllModelParameters) with identity added to scale parameters,
         model_identity is the identity/scale parameters as a numpy array of shape (numAllModelParameters,),
         and fps is the frames per second of the motion data.
)",
          py::arg("gltf_bytes"))
      // toGLTF(character, fps, motion)
      .def_static(
          "to_gltf",
          &toGLTF,
          py::call_guard<py::gil_scoped_release>(),
          R"(Serialize a character as a GLTF using dictionary form.

:param character: A valid character.
:param fps: Frames per second for describing the motion.
:param motion: tuple of vector of parameter names and a P X T matrix. P is number of parameters, T is number of frames.
:param offsets: tuple of vector of joint names and a Vector of size J * 7 (Parameters per joint). Eg. for 3 joints, you would have 21 params.
:return: a GLTF representation of Character with motion
      )",
          py::arg("character"))
      // loadFBXCharacterFromFile(fbxFilename, modelFilename, locatorsFilename)
      .def_static(
          "load_fbx",
          &loadFBXCharacterFromFile,
          py::call_guard<py::gil_scoped_release>(),
          R"(Load a character from an FBX file.  Optionally pass in a separate model definition and locators file.

:param fbx_filename: .fbx file that contains the skeleton and skinned mesh; e.g. character_s0.fbx.
:param model_filename: Configuration file that defines the parameter mappings and joint limits; e.g. character.cfg.
:param locators_filename: File containing the locators, e.g. character.locators.
:param permissive: If true, ignore certain errors during loading.
:param load_blendshapes: If true, load blendshapes from the file.
:param strip_namespaces: If true, strip namespaces from nodes
:return: A valid :class:`Character`.)",
          py::arg("fbx_filename"),
          py::arg("model_filename") = std::optional<std::string>{},
          py::arg("locators_filename") = std::optional<std::string>{},
          py::arg("permissive") = false,
          py::arg("load_blendshapes") = false,
          py::arg("strip_namespaces") = true)
      // loadFBXCharacterFromFileWithMotion(fbxFilename, modelFilename,
      // locatorsFilename)
      .def_static(
          "load_fbx_with_motion",
          &loadFBXCharacterWithMotionFromFile,
          py::call_guard<py::gil_scoped_release>(),
          R"(Load a character and animation curves from an FBX file.

:param fbx_filename: .fbx file that contains the skeleton and skinned mesh; e.g. character_s0.fbx.
:param permissive: If true, ignore certain errors during loading.
:param load_blendshapes: If true, load blendshapes from the file.
:param strip_namespaces: If true, strip namespaces from nodes
:return: A valid :class:`Character`, a vector of motions in the format of nFrames X nNumJointParameters, and fps. The caller needs to decide how to handle the joint parameters.)",
          py::arg("fbx_filename"),
          py::arg("permissive") = false,
          py::arg("load_blendshapes") = false,
          py::arg("strip_namespaces") = true)

      .def_static(
          "load_fbx_with_motion_from_bytes",
          &loadFBXCharacterWithMotionFromBytes,
          py::call_guard<py::gil_scoped_release>(),
          R"(Load a character and animation curves from an FBX file.

:param fbx_bytes: A Python bytes that is an .fbx file containing the skeleton and skinned mesh.
:param permissive: If true, ignore certain errors during loading.
:param load_blendshapes: If true, load blendshapes from the file.
:param strip_namespaces: If true, strip namespaces from nodes
:return: A valid :class:`Character`, a vector of motions in the format of nFrames X nNumJointParameters, and fps. The caller needs to decide how to handle the joint parameters.)",
          py::arg("fbx_bytes"),
          py::arg("permissive") = false,
          py::arg("load_blendshapes") = false,
          py::arg("strip_namespaces") = true)

      // loadFBXCharacterFromBytes(fbxBytes)
      .def_static(
          "load_fbx_from_bytes",
          &pymomentum::loadFBXCharacterFromBytes,
          py::call_guard<py::gil_scoped_release>(),
          R"(Load a character from byte array for an FBX file.

:param fbx_bytes: An array of bytes in FBX format.
:param permissive: If true, ignore certain errors during loading.
:param load_blendshapes: If true, load blendshapes from the file.
:param strip_namespaces: If true, strip namespaces from nodes
:return: A valid :class:`Character`.)",
          py::arg("fbx_bytes"),
          py::arg("permissive") = false,
          py::arg("load_blendshapes") = false,
          py::arg("strip_namespaces") = true)
      // loadLocatorsFromFile(character, locatorFile)
      .def(
          "load_locators",
          &pymomentum::loadLocatorsFromFile,
          py::call_guard<py::gil_scoped_release>(),
          R"(Load locators from a .locators file.

:param character: The character to map the locators onto.
:param filename: Filename for the locators.
:return: A valid :class:`Character`.)",
          py::arg("filename"))
      // loadLocatorsFromBytes(character, locatorBytes)
      .def(
          "load_locators_from_bytes",
          &pymomentum::loadLocatorsFromBytes,
          py::call_guard<py::gil_scoped_release>(),
          R"(Load locators from a byte array containing .locators file data.

:param character: The character to map the locators onto.
:param locator_bytes: A byte array containing the locators.
:return: A valid :class:`Character`.)",
          py::arg("locator_bytes"))
      // localModelDefinitionFromFile(character, modelFile)
      .def(
          "load_model_definition",
          &pymomentum::loadConfigFromFile,
          py::call_guard<py::gil_scoped_release>(),
          R"(Load a model definition from a .model file.  This defines the parameter transform, model parameters, and joint limits.

:param character: The character containing a valid skeleton.
:param filename: Filename for the model definition.
:return: A valid :class:`Character`.)",
          py::arg("filename"))
      // localModelDefinitionFromBytes(character, modelBytes)
      .def(
          "load_model_definition_from_bytes",
          &pymomentum::loadConfigFromBytes,
          py::call_guard<py::gil_scoped_release>(),
          R"(Load a model definition from a .model file.  This defines the parameter transform, model parameters, and joint limits.

:param character: The character containing a valid skeleton.
:param model_bytes: Bytes array containing the model definition.
:return: A valid :class:`Character`.)",
          py::arg("model_bytes"))
      // loadCharacterWithMotion(gltfFilename)
      .def_static(
          "load_gltf_with_motion",
          &loadGLTFCharacterWithMotion,
          py::call_guard<py::gil_scoped_release>(),
          R"(Load both character (skeleton/mesh) and motion data from a gltf file in a single call.

Use this function when you need both the character structure and its animation together.
For loading ONLY motion data without the character, use :meth:`pymomentum.geometry.load_motion` instead.

Note that motion can only be read from GLTF files saved using momentum, which stores model parameters
in a custom extension. For GLTF files saved using other software, use :meth:`load_gltf_with_skel_states`.

:param gltf_filename: A .gltf file; e.g. character_s0.glb.
:return: a tuple [Character, motion, identity, fps], where Character is the complete character object,
         motion is the motion matrix [nFrames x nParams], identity is a JointParameter at rest pose,
         and fps is the frame rate. Does NOT include parameter names (unlike :meth:`pymomentum.geometry.load_motion`).
      )",
          py::arg("gltf_filename"))
      .def_static(
          "load_gltf_with_skel_states_from_bytes",
          &loadGLTFCharacterWithSkelStatesFromBytes,
          R"(Load a character and a skeleton state motion sequence from gltf bytes.  Unlike
:meth:`load_gltf_with_motion`, this function should work with any GLTF file since it reads the raw transforms from the file
and doesn't require that the Character have a valid parameter transform.  Unlike :meth:`load_gltf_with_motion`, it does not
support the proprietary momentum motion format for storing model parameters in GLB.

:param gltf_bytes: The bytes of a gltf file.
:return: a tuple [Character, skel_states, fps], where skel_states is the tensor [nFrames x nJoints x 8].
        )",
          py::arg("gltf_bytes"))
      .def_static(
          "load_gltf_with_skel_states",
          &loadGLTFCharacterWithSkelStates,
          R"(Load a character and a skel state sequence from a gltf file.  Unlike
:meth:`load_gltf_with_motion`, this function should work with any GLTF file since it reads the raw transforms from the file
and doesn't require that the Character have a valid parameter transform.  Unlike :meth:`load_gltf_with_motion`, it does not
support the proprietary momentum motion format for storing model parameters in GLB.

:param gltf_filename: A .gltf file; e.g. character_s0.glb.
:return: a tuple [Character, skel_states, timestamps], where skel_states is the tensor [n_frames x n_joints x 8] and timestamps is [n_frames]
          )",
          py::arg("gltf_filename"))
      .def_static(
          "load_motion_timestamps",
          &loadMotionTimestamps,
          R"(Load per-frame timestamps from a gltf file.

Load timestamps stored in the momentum extension (usually in microseconds).

:param gltf_filename: A .gltf file; e.g. character_s0.glb.
:return: A list of timestamps (int64), one per frame. Empty list if no timestamps found.
      )",
          py::arg("gltf_filename"))

      // loadGLTFCharacterFromFile(filename)
      .def_static(
          "load_gltf",
          &loadGLTFCharacterFromFile,
          py::call_guard<py::gil_scoped_release>(),
          R"(Load a character from a gltf file.

:param path: A .gltf file; e.g. character_s0.glb.
      )",
          py::arg("path"))
      // loadURDFCharacterFromFile(urdfPath)
      .def_static(
          "load_urdf",
          &loadURDFCharacterFromFile,
          py::call_guard<py::gil_scoped_release>(),
          R"(Load a character from a urdf file.

:param urdf_filename: A .urdf file; e.g. character.urdf.
      )",
          py::arg("urdf_filename"))
      // loadURDFCharacterFromBytes(urdfBytes)
      .def_static(
          "load_urdf_from_bytes",
          &loadURDFCharacterFromBytes,
          py::call_guard<py::gil_scoped_release>(),
          R"(Load a character from urdf bytes.

:param urdf_bytes: Bytes array containing the urdf definition.
      )",
          py::arg("urdf_bytes"))
#ifdef MOMENTUM_WITH_USD
      // loadUSDCharacterFromFile(usdPath)
      .def_static(
          "load_usd",
          &loadUSDCharacterFromFile,
          py::call_guard<py::gil_scoped_release>(),
          R"(Load a character from a USD file.

Supports .usd, .usda, .usdc, and .usdz file formats.

:param usd_filename: Path to the USD file.
:return: A :class:`Character` object containing the loaded skeleton, mesh, and skin weights.
      )",
          py::arg("usd_filename"))
      // loadUSDCharacterFromBytes(usdBytes)
      .def_static(
          "load_usd_from_bytes",
          &loadUSDCharacterFromBytes,
          py::call_guard<py::gil_scoped_release>(),
          R"(Load a character from USD bytes.

:param usd_bytes: Bytes array containing the USD data.
:return: A :class:`Character` object containing the loaded skeleton, mesh, and skin weights.
      )",
          py::arg("usd_bytes"))
      // saveUSDCharacterToFile(path, character)
      .def_static(
          "save_usd",
          &saveUSDCharacterToFile,
          py::call_guard<py::gil_scoped_release>(),
          R"(Save a character to a USD file.

:param path: Path to save the USD file.
:param character: The :class:`Character` object to save.
      )",
          py::arg("path"),
          py::arg("character"))
#endif // MOMENTUM_WITH_USD
      // saveGLTFCharacterToFile(filename, character)
      .def_static(
          "save_gltf",
          &saveGLTFCharacterToFile,
          py::call_guard<py::gil_scoped_release>(),
          R"(Save a character to a gltf file.

:param path: A .gltf export filename.
:param character: A Character to be saved to the output file.
:param fps: Frequency in frames per second
:param motion: Pose array in [n_frames x n_parameters]
:param offsets: Offset array in [n_joints x n_parameters_per_joint]
:param markers: Additional marker (3d positions) data in [n_frames][n_markers]
:param options: FileSaveOptions for controlling output (mesh, locators, collisions, etc.)
:param timestamps: Per-frame timestamps (usually in microseconds). Size should match the number of motion frames.
      )",
          py::arg("path"),
          py::arg("character"),
          py::arg("fps") = 120.f,
          py::arg("motion") = std::nullopt,
          py::arg("offsets") = std::nullopt,
          py::arg("markers") = std::nullopt,
          py::arg("options") = std::nullopt,
          py::arg("timestamps") = std::nullopt)
      .def_static(
          "save_gltf_from_skel_states",
          &saveGLTFCharacterToFileFromSkelStates,
          py::call_guard<py::gil_scoped_release>(),
          R"(Save a character to a gltf file.

:param path: A .gltf export filename.
:param character: A Character to be saved to the output file.
:param fps: Frequency in frames per second
:param skel_states: Skeleton states [n_frames x n_joints x n_parameters_per_joint]
:param markers: Additional marker (3d positions) data in [n_frames][n_markers]
:param options: FileSaveOptions for controlling output (mesh, locators, collisions, etc.)
      )",
          py::arg("path"),
          py::arg("character"),
          py::arg("fps"),
          py::arg("skel_states"),
          py::arg("markers") = std::optional<const std::vector<std::vector<momentum::Marker>>>{},
          py::arg("options") = std::optional<momentum::FileSaveOptions>{})
      .def_static(
          "save_fbx",
          &saveFBXCharacterToFile,
          py::call_guard<py::gil_scoped_release>(),
          R"(Save a character to an fbx file.

:param path: An .fbx export filename.
:param character: A Character to be saved to the output file.
:param fps: Frequency in frames per second
:param motion: [Optional] 2D pose matrix in [n_frames x n_parameters]
:param offsets: [Optional] Offset array in [(n_joints x n_parameters_per_joint)]
:param markers: Additional marker (3d positions) data in [n_frames][n_markers]
:param options: [Optional] FileSaveOptions for controlling output (mesh, locators, collisions, coordinate system, namespace, etc.)
      )",
          py::arg("path"),
          py::arg("character"),
          py::arg("fps") = 120.f,
          py::arg("motion") = std::optional<const Eigen::MatrixXf>{},
          py::arg("offsets") = std::optional<const Eigen::VectorXf>{},
          py::arg("markers") = std::optional<const std::vector<std::vector<momentum::Marker>>>{},
          py::arg("options") = momentum::FileSaveOptions())
      .def_static(
          "save_fbx_with_joint_params",
          &saveFBXCharacterToFileWithJointParams,
          py::call_guard<py::gil_scoped_release>(),
          R"(Save a character to an fbx file with joint params.

:param path: An .fbx export filename.
:param character: A Character to be saved to the output file.
:param fps: Frequency in frames per second
:param joint_params: [Optional] 2D pose matrix in [n_frames x n_parameters]
:param markers: Additional marker (3d positions) data in [n_frames][n_markers]
:param options: [Optional] FileSaveOptions for controlling output (mesh, locators, collisions, coordinate system, namespace, etc.)
      )",
          py::arg("path"),
          py::arg("character"),
          py::arg("fps") = 120.f,
          py::arg("joint_params") = std::optional<const Eigen::MatrixXf>{},
          py::arg("markers") = std::optional<const std::vector<std::vector<momentum::Marker>>>{},
          py::arg("options") = momentum::FileSaveOptions())
      .def_static(
          "save",
          &saveCharacterToFile,
          py::call_guard<py::gil_scoped_release>(),
          R"(Save a character to a file. The format is determined by the file extension (.fbx, .glb, .gltf).

    This is a unified interface that automatically selects between FBX and GLTF based on the file extension.

:param path: Export filename with extension (.fbx, .glb, or .gltf).
:param character: A Character to be saved to the output file.
:param fps: [Optional] Frequency in frames per second
:param motion: [Optional] 2D pose matrix in [n_frames x n_parameters]
:param offsets: [Optional] Offset array in [(n_joints x n_parameters_per_joint)]
:param markers: [Optional] Additional marker (3d positions) data in [n_frames][n_markers]
:param options: [Optional] FileSaveOptions for controlling output (mesh, locators, collisions, etc.)
      )",
          py::arg("path"),
          py::arg("character"),
          py::arg("fps") = 120.f,
          py::arg("motion") = std::optional<const Eigen::MatrixXf>{},
          py::arg("markers") = std::optional<const std::vector<std::vector<momentum::Marker>>>{},
          py::arg("options") = momentum::FileSaveOptions())
      .def_static(
          "save_with_skel_states",
          &saveCharacterToFileWithSkelStates,
          py::call_guard<py::gil_scoped_release>(),
          R"(Save a character to a file using skeleton states. The format is determined by the file extension (.fbx, .glb, .gltf).

    This function allows saving a character and its animation using skeleton state matrices instead of model parameters.

    :param path: Export filename with extension (.fbx, .glb, or .gltf).
    :param character: A Character to be saved to the output file.
    :param fps: Frequency in frames per second
    :param skel_states: Skeleton states [n_frames x n_joints x n_parameters_per_joint]
    :param markers: [Optional] Additional marker (3d positions) data in [n_frames][n_markers]
    )",
          py::arg("path"),
          py::arg("character"),
          py::arg("fps"),
          py::arg("skel_states"),
          py::arg("markers") = std::optional<const std::vector<std::vector<momentum::Marker>>>{})
      // Legacy JSON I/O methods
      .def_static(
          "load_legacy_json",
          [](const std::string& jsonPath) { return mm::loadCharacterFromLegacyJson(jsonPath); },
          py::call_guard<py::gil_scoped_release>(),
          R"(Load a character from a legacy JSON file.

This function directly converts from the deprecated JSON format to momentum::Character,
this is a legacy format that has historically been used in previous Python libraries but should be considered deprecated.

:param json_path: Path to the legacy JSON file.
:return: A valid Character.)",
          py::arg("json_path"))
      .def_static(
          "load_legacy_json_from_bytes",
          [](const py::bytes& jsonBytes) {
            std::string jsonString(jsonBytes);
            std::span<const std::byte> buffer(
                reinterpret_cast<const std::byte*>(jsonString.data()), jsonString.size());
            return mm::loadCharacterFromLegacyJsonBuffer(buffer);
          },
          py::call_guard<py::gil_scoped_release>(),
          R"(Load a character from legacy JSON bytes.

:param json_bytes: A bytes object containing the legacy JSON data.
:return: A valid Character.)",
          py::arg("json_bytes"))
      .def_static(
          "load_legacy_json_from_string",
          [](const std::string& jsonString) {
            return mm::loadCharacterFromLegacyJsonString(jsonString);
          },
          py::call_guard<py::gil_scoped_release>(),
          R"(Load a character from a legacy JSON string.

:param json_string: String containing the legacy JSON data.
:return: A valid Character.)",
          py::arg("json_string"))
      .def_static(
          "save_legacy_json",
          [](const mm::Character& character, const std::string& jsonPath) {
            mm::saveCharacterToLegacyJson(character, jsonPath);
          },
          py::call_guard<py::gil_scoped_release>(),
          R"(Save a character to legacy JSON format.

This function converts a momentum::Character back to the legacy JSON format
for compatibility with existing tools and workflows.

:param character: The Character to save.
:param json_path: Path where to save the legacy JSON file.)",
          py::arg("character"),
          py::arg("json_path"))
      .def_static(
          "to_legacy_json_string",
          [](const mm::Character& character) { return mm::characterToLegacyJsonString(character); },
          py::call_guard<py::gil_scoped_release>(),
          R"(Convert a character to legacy JSON string.

:param character: The Character to convert.
:return: String containing the legacy JSON representation.)",
          py::arg("character"))
      .def(
          "simplify",
          [](const momentum::Character& character,
             std::optional<at::Tensor> enabledParamsTensor) -> momentum::Character {
            momentum::ParameterSet enabledParams;
            if (enabledParamsTensor) {
              enabledParams =
                  tensorToParameterSet(character.parameterTransform, *enabledParamsTensor);
            } else {
              enabledParams.set();
            }
            return character.simplify(enabledParams);
          },
          R"(Simplifies the character by removing extra joints; this can help to speed up IK, but passing in a set of
parameters rather than joints.  Does not modify the parameter transform.  This is the equivalent of calling
```character.simplify_skeleton(character.joints_from_parameters(enabled_params))```.

:param enabled_parameters: Model parameters to be kept in the simplified model.  Defaults to including all parameters.
:return: a new :class:`Character` with extraneous joints removed.)",
          py::arg("enabled_parameters") = std::optional<at::Tensor>{})
      .def(
          "simplify_skeleton",
          [](const momentum::Character& character,
             const std::vector<int>& enabledJointIndices) -> momentum::Character {
            return character.simplifySkeleton(jointListToBitset(character, enabledJointIndices));
          },
          "Simplifies the character by removing unwanted joints.",
          py::arg("enabled_joint_indices"))
      .def(
          "simplify_parameter_transform",
          [](const momentum::Character& character,
             at::Tensor enabledParameters) -> momentum::Character {
            return character.simplifyParameterTransform(
                tensorToParameterSet(character.parameterTransform, enabledParameters));
          },
          "Simplifies the character by removing unwanted parameters.",
          py::arg("enabled_parameters"))
      .def(
          "parameters_for_joints",
          [](const momentum::Character& character, const std::vector<int>& jointIndices) {
            return parameterSetToTensor(
                character.parameterTransform,
                character.activeJointsToParameters(jointListToBitset(character, jointIndices)));
          },
          "Maps a list of joint indices to a boolean tensor containing the parameters which drive those joints.",
          py::arg("joint_indices"))
      .def(
          "joints_for_parameters",
          [](const momentum::Character& character, at::Tensor enabledParamsTensor) {
            return bitsetToJointList(character.parametersToActiveJoints(
                tensorToParameterSet(character.parameterTransform, enabledParamsTensor)));
          },
          "Maps a list of parameter indices to a list of joints driven by those parameters.",
          py::arg("active_parameters"))
      .def("__repr__", [](const mm::Character& c) {
        return fmt::format(
            "Character(name='{}', joints={}, parameters={}, has_mesh={})",
            c.name,
            c.skeleton.joints.size(),
            c.parameterTransform.numAllModelParameters(),
            c.mesh ? "True" : "False");
      });
}

} // namespace pymomentum
