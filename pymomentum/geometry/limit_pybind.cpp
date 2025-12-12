/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/geometry/limit_pybind.h"

#include <momentum/character/parameter_limits.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Core>

#include <limits>
#include <optional>

#include <fmt/format.h>

namespace py = pybind11;
namespace mm = momentum;

namespace pymomentum {

namespace {

std::string parameterLimitRepr(const mm::ParameterLimit& pl) {
  std::string typeStr;
  std::string dataStr;

  switch (pl.type) {
    case mm::LimitType::MinMax:
      typeStr = "MinMax";
      dataStr = fmt::format(
          "param={}, min={}, max={}",
          pl.data.minMax.parameterIndex,
          pl.data.minMax.limits[0],
          pl.data.minMax.limits[1]);
      break;
    case mm::LimitType::MinMaxJoint:
      typeStr = "MinMaxJoint";
      dataStr = fmt::format(
          "joint={}, param={}, min={}, max={}",
          pl.data.minMaxJoint.jointIndex,
          pl.data.minMaxJoint.jointParameter,
          pl.data.minMaxJoint.limits[0],
          pl.data.minMaxJoint.limits[1]);
      break;
    case mm::LimitType::MinMaxJointPassive:
      typeStr = "MinMaxJointPassive";
      dataStr = fmt::format(
          "joint={}, param={}, min={}, max={}",
          pl.data.minMaxJoint.jointIndex,
          pl.data.minMaxJoint.jointParameter,
          pl.data.minMaxJoint.limits[0],
          pl.data.minMaxJoint.limits[1]);
      break;
    case mm::LimitType::Linear:
      typeStr = "Linear";
      dataStr = fmt::format(
          "ref={}, target={}, scale={}, offset={}",
          pl.data.linear.referenceIndex,
          pl.data.linear.targetIndex,
          pl.data.linear.scale,
          pl.data.linear.offset);
      break;
    case mm::LimitType::LinearJoint:
      typeStr = "LinearJoint";
      dataStr = fmt::format(
          "ref_joint={}, ref_param={}, target_joint={}, target_param={}, scale={}, offset={}",
          pl.data.linearJoint.referenceJointIndex,
          pl.data.linearJoint.referenceJointParameter,
          pl.data.linearJoint.targetJointIndex,
          pl.data.linearJoint.targetJointParameter,
          pl.data.linearJoint.scale,
          pl.data.linearJoint.offset);
      break;
    case mm::LimitType::Ellipsoid:
      typeStr = "Ellipsoid";
      dataStr = fmt::format(
          "ellipsoid_parent={}, parent={}, offset=[{} {} {}]",
          pl.data.ellipsoid.ellipsoidParent,
          pl.data.ellipsoid.parent,
          pl.data.ellipsoid.offset[0],
          pl.data.ellipsoid.offset[1],
          pl.data.ellipsoid.offset[2]);
      break;
    case mm::LimitType::HalfPlane:
      typeStr = "HalfPlane";
      dataStr = fmt::format(
          "param1={}, param2={}, normal=[{} {}], offset={}",
          pl.data.halfPlane.param1,
          pl.data.halfPlane.param2,
          pl.data.halfPlane.normal[0],
          pl.data.halfPlane.normal[1],
          pl.data.halfPlane.offset);
      break;
    default:
      typeStr = "Unknown";
      dataStr = "";
      break;
  }

  if (!dataStr.empty()) {
    return fmt::format("ParameterLimit(type={}, weight={}, {})", typeStr, pl.weight, dataStr);
  } else {
    return fmt::format("ParameterLimit(type={}, weight={})", typeStr, pl.weight);
  }
}

} // namespace

void registerLimits(py::module_& m, py::class_<mm::ParameterLimit>& parameterLimitClass) {
  py::enum_<mm::LimitType>(m, "LimitType", R"(Type of joint limit.)")
      .value("MinMax", mm::LimitType::MinMax)
      .value("MinMaxJoint", mm::LimitType::MinMaxJoint)
      .value("MinMaxJointPassive", mm::LimitType::MinMaxJointPassive)
      .value("Linear", mm::LimitType::Linear)
      .value("LinearJoint", mm::LimitType::LinearJoint)
      .value("Ellipsoid", mm::LimitType::Ellipsoid)
      .value("HalfPlane", mm::LimitType::HalfPlane);

  auto parameterLimitDataClass = py::class_<mm::LimitData>(
      m,
      "LimitData",
      "Data container for parameter limits. Contains the specific constraint data "
      "for different limit types (MinMax, Linear, Ellipsoid, etc.).");
  auto parameterLimitMinMaxClass = py::class_<mm::LimitMinMax>(
      m,
      "LimitMinMax",
      "Min/max constraint data for model parameters. Contains the parameter index "
      "and the minimum and maximum allowed values.");
  auto parameterLimitMinMaxJointClass = py::class_<mm::LimitMinMaxJoint>(
      m,
      "LimitMinMaxJoint",
      "Min/max constraint data for joint parameters. Contains the joint index, "
      "joint parameter index, and the minimum and maximum allowed values.");
  auto parameterLimitLinearClass = py::class_<mm::LimitLinear>(
      m,
      "LimitLinear",
      "Linear constraint data for model parameters. Enforces a linear relationship "
      "between two parameters of the form: p0 = scale * p1 + offset.");
  auto parameterLimitLinearJointClass = py::class_<mm::LimitLinearJoint>(
      m,
      "LimitLinearJoint",
      "Linear constraint data for joint parameters. Enforces a linear relationship "
      "between two joint parameters of the form: p0 = scale * p1 + offset.");
  auto parameterLimitHalfPlaneClass = py::class_<mm::LimitHalfPlane>(
      m,
      "LimitHalfPlane",
      "Half-plane constraint data for model parameters. Enforces that parameters "
      "lie on one side of a plane defined by a normal vector and offset.");
  auto parameterLimitEllipsoidClass = py::class_<mm::LimitEllipsoid>(
      m,
      "LimitEllipsoid",
      "Ellipsoid constraint data for model parameters. Enforces that parameters "
      "lie within an ellipsoid defined by a transformation matrix and offset.");

  parameterLimitClass.def_readonly("type", &mm::ParameterLimit::type, "Type of parameter limit.")
      .def_readonly("weight", &mm::ParameterLimit::weight, "Weight of parameter limit.")
      .def_readonly("data", &mm::ParameterLimit::data, "Data of parameter limit.")
      .def_static(
          "create_minmax",
          [](size_t model_parameter_index, float min, float max, float weight) {
            mm::LimitData data;
            data.minMax.parameterIndex = model_parameter_index;
            data.minMax.limits = Eigen::Vector2f(min, max);
            return mm::ParameterLimit{data, mm::LimitType::MinMax, weight};
          },
          R"(
Create a parameter limit with min and max values for a model parameter.

:parameter model_parameter_index: Index of model parameter to limit.
:parameter min: Minimum value of the parameter.
:parameter max: Maximum value of the parameter.
:parameter weight: Weight of the parameter limit.  Defaults to 1.
        )",
          py::arg("model_parameter_index"),
          py::arg("min"),
          py::arg("max"),
          py::arg("weight") = 1.0f)
      .def_static(
          "create_minmax_joint",
          [](size_t joint_index, size_t joint_parameter, float min, float max, float weight) {
            mm::LimitData data;
            data.minMaxJoint.jointIndex = joint_index;
            data.minMaxJoint.jointParameter = joint_parameter;
            data.minMaxJoint.limits = Eigen::Vector2f(min, max);
            return mm::ParameterLimit{data, mm::LimitType::MinMaxJoint, weight};
          },
          R"(
Create a parameter limit with min and max values for a joint parameter.

:parameter joint_index: Index of joint to limit.
:parameter joint_parameter: Index of joint parameter to limit, in the range 0->7 (tx,ty,tz,rx,ry,rz,s).
:parameter min: Minimum value of the parameter.
:parameter max: Maximum value of the parameter.
:parameter weight: Weight of the parameter limit.  Defaults to 1.
        )",
          py::arg("joint_index"),
          py::arg("joint_parameter"),
          py::arg("min"),
          py::arg("max"),
          py::arg("weight") = 1.0f)
      .def_static(
          "create_linear",
          [](size_t reference_model_parameter_index,
             size_t target_model_parameter_index,
             float scale,
             float offset,
             float weight,
             std::optional<float> rangeMin,
             std::optional<float> rangeMax) {
            mm::LimitData data;
            data.linear.referenceIndex = reference_model_parameter_index;
            data.linear.targetIndex = target_model_parameter_index;
            data.linear.scale = scale;
            data.linear.offset = offset;
            data.linear.rangeMin = rangeMin.value_or(-std::numeric_limits<float>::max());
            data.linear.rangeMax = rangeMax.value_or(std::numeric_limits<float>::max());
            return mm::ParameterLimit{data, mm::LimitType::Linear, weight};
          },
          R"(Create a parameter limit with a linear constraint.

:parameter reference_model_parameter_index: Index of reference parameter p0 to use in equation p_0 = scale * p_1 - offset.
:parameter target_model_parameter_index: Index of target parameter p1 to use in equation p_0 = scale * p_1 - offset.
:parameter scale: Scale to use in equation p_0 = scale * p_1 - offset.
:parameter offset: Offset to use in equation p_0 = scale * p_1 - offset.
:parameter weight: Weight of the parameter limit.  Defaults to 1.
:parameter range_min: Minimum of the range that the linear limit applies over.  Defaults to -infinity.
:parameter range_max: Minimum of the range that the linear limit applies over.  Defaults to +infinity.
    )",
          py::arg("reference_model_parameter_index"),
          py::arg("target_model_parameter_index"),
          py::arg("scale"),
          py::arg("offset"),
          py::arg("weight") = 1.0f,
          py::arg("range_min") = std::optional<float>{},
          py::arg("range_max") = std::optional<float>{})
      .def_static(
          "create_linear_joint",
          [](size_t reference_joint_index,
             size_t reference_joint_parameter,
             size_t target_joint_index,
             size_t target_joint_parameter,
             float scale,
             float offset,
             float weight,
             std::optional<float> rangeMin,
             std::optional<float> rangeMax) {
            mm::LimitData data;
            data.linearJoint.referenceJointIndex = reference_joint_index;
            data.linearJoint.referenceJointParameter = reference_joint_parameter;
            data.linearJoint.targetJointIndex = target_joint_index;
            data.linearJoint.targetJointParameter = target_joint_parameter;
            data.linearJoint.scale = scale;
            data.linearJoint.offset = offset;
            data.linearJoint.rangeMin = rangeMin.value_or(-std::numeric_limits<float>::max());
            data.linearJoint.rangeMax = rangeMax.value_or(std::numeric_limits<float>::max());
            return mm::ParameterLimit{data, mm::LimitType::LinearJoint, weight};
          },
          R"(Create a parameter limit with a linear joint constraint.

:parameter reference_joint_index: Index of reference joint p0 to use in equation p_0 = scale * p_1 - offset.
:parameter reference_joint_parameter: Index of parameter within joint to use.
:parameter target_joint_index: Index of target parameter p1 to use in equation p_0 = scale * p_1 - offset.
:parameter target_joint_parameter: Index of parameter within joint to use.
:parameter scale: Scale to use in equation p_0 = scale * p_1 - offset.
:parameter offset: Offset to use in equation p_0 = scale * p_1 - offset.
:parameter weight: Weight of the parameter limit.  Defaults to 1.
:parameter range_min: Minimum of the range that the linear limit applies over.  Defaults to -infinity.
:parameter range_max: Minimum of the range that the linear limit applies over.  Defaults to +infinity.
    )",
          py::arg("reference_joint_index"),
          py::arg("reference_joint_parameter"),
          py::arg("target_joint_index"),
          py::arg("target_joint_parameter"),
          py::arg("scale"),
          py::arg("offset"),
          py::arg("weight") = 1.0f,
          py::arg("range_min") = std::optional<float>{},
          py::arg("range_max") = std::optional<float>{})
      .def_static(
          "create_halfplane",
          [](size_t param1, size_t param2, Eigen::Vector2f normal, float offset, float weight) {
            mm::LimitData data;
            data.halfPlane.param1 = param1;
            data.halfPlane.param2 = param2;

            const float len = normal.norm();
            data.halfPlane.normal = normal / len;
            data.halfPlane.offset = offset / len;
            return mm::ParameterLimit{data, mm::LimitType::HalfPlane, weight};
          },
          R"(Create a parameter limit with a half-plane constraint.

:parameter param1_index: Index of the first parameter in the plane equation (p1, p2) . (n1, n2) - offset >= 0.
:parameter param2_index: Index of the second parameter (p1, p2) . (n1, n2) - offset >= 0.
:parameter offset: Offset to use in equation (p1, p2) . (n1, n2) - offset >= 0.
:parameter weight: Weight of the parameter limit.  Defaults to 1.
    )",
          py::arg("param1_index"),
          py::arg("param2_index"),
          py::arg("normal"),
          py::arg("offset") = 0.0f,
          py::arg("weight") = 1.0f)
      .def_static(
          "create_ellipsoid",
          [](size_t ellipsoid_parent,
             size_t parent,
             const Eigen::Vector3f& offset,
             const Eigen::Matrix4f& ellipsoid,
             float weight) {
            mm::LimitData data;
            data.ellipsoid.ellipsoidParent = ellipsoid_parent;
            data.ellipsoid.parent = parent;
            data.ellipsoid.offset = offset;
            data.ellipsoid.ellipsoid = Eigen::Affine3f(ellipsoid);
            data.ellipsoid.ellipsoidInv = data.ellipsoid.ellipsoid.inverse();
            return mm::ParameterLimit{data, mm::LimitType::Ellipsoid, weight};
          },
          R"(Create a parameter limit with an ellipsoid constraint.

:parameter ellipsoid_parent: Index of joint to use as the ellipsoid's parent.
:parameter parent: Index of joint to constraint.
:parameter offset: Offset of the ellipsoid from the parent joint.
:parameter ellipsoid: 4x4 matrix defining the ellipsoid's shape.
:parameter weight: Weight of the parameter limit.  Defaults to 1.
    )",
          py::arg("ellipsoid_parent"),
          py::arg("parent"),
          py::arg("offset"),
          py::arg("ellipsoid"),
          py::arg("weight") = 1.0f)
      .def("__repr__", &parameterLimitRepr);

  parameterLimitDataClass.def_readonly("minmax", &mm::LimitData::minMax, "Data for MinMax limit.")
      .def_readonly("minmax_joint", &mm::LimitData::minMaxJoint, "Data for MinMaxJoint limit.")
      .def_readonly("linear", &mm::LimitData::linear, "Data for Linear limit.")
      .def_readonly("linear_joint", &mm::LimitData::linearJoint, "Data for LinearJoint limit.")
      .def_readonly("halfplane", &mm::LimitData::halfPlane, "Data for HalfPlane limit.")
      .def_readonly("ellipsoid", &mm::LimitData::ellipsoid, "Data for Ellipsoid limit.")
      .def("__repr__", []([[maybe_unused]] const mm::LimitData& ld) {
        return fmt::format("LimitData()");
      });

  parameterLimitMinMaxClass
      .def_readonly(
          "model_parameter_index",
          &mm::LimitMinMax::parameterIndex,
          "Index of model parameter to use.")
      .def_property_readonly(
          "min",
          [](const mm::LimitMinMax& data) { return data.limits[0]; },
          "Minimum value of MinMax limit.")
      .def_property_readonly(
          "max",
          [](const mm::LimitMinMax& data) { return data.limits[1]; },
          "Maximum value of MinMax limit.")
      .def("__repr__", [](const mm::LimitMinMax& lmm) {
        return fmt::format(
            "LimitMinMax(param={}, min={}, max={})",
            lmm.parameterIndex,
            lmm.limits[0],
            lmm.limits[1]);
      });

  parameterLimitMinMaxJointClass
      .def_readonly("joint_index", &mm::LimitMinMaxJoint::jointIndex, "Index of joint to affect.")
      .def_readonly(
          "joint_parameter_index",
          &mm::LimitMinMaxJoint::jointParameter,
          "Index of joint parameter to use, in the range 0->7 (tx,ty,tz,rx,ry,rz,s).")
      .def_property_readonly(
          "min",
          [](const mm::LimitMinMaxJoint& data) { return data.limits[0]; },
          "Minimum value of MinMaxJoint limit.")
      .def_property_readonly(
          "max",
          [](const mm::LimitMinMaxJoint& data) { return data.limits[1]; },
          "Maximum value of MinMaxJoint limit.")
      .def("__repr__", [](const mm::LimitMinMaxJoint& lmmj) {
        return fmt::format(
            "LimitMinMaxJoint(joint={}, param={}, min={}, max={})",
            lmmj.jointIndex,
            lmmj.jointParameter,
            lmmj.limits[0],
            lmmj.limits[1]);
      });

  parameterLimitLinearClass
      .def_readonly(
          "reference_model_parameter_index",
          &mm::LimitLinear::referenceIndex,
          "Index of reference parameter p0 to use in equation p_0 = scale * p_1 - offset.")
      .def_readonly(
          "target_model_parameter_index",
          &mm::LimitLinear::targetIndex,
          "Index of target parameter p1 to use in equation p_0 = scale * p_1 - offset.")
      .def_readonly(
          "scale", &mm::LimitLinear::scale, "Scale to use in equation p_0 = scale * p_1 - offset.")
      .def_readonly(
          "offset",
          &mm::LimitLinear::offset,
          "Offset to use in equation p_0 = scale * p_1 - offset.")
      .def_property_readonly(
          "range_min",
          [](const mm::LimitLinear& data) -> std::optional<float> {
            if (data.rangeMin <= -std::numeric_limits<float>::max()) {
              return std::optional<float>{};
            } else {
              return std::make_optional(data.rangeMin);
            }
          })
      .def_property_readonly(
          "range_max",
          [](const mm::LimitLinear& data) -> std::optional<float> {
            if (data.rangeMax >= std::numeric_limits<float>::max()) {
              return std::optional<float>{};
            } else {
              return std::make_optional(data.rangeMax);
            }
          })
      .def("__repr__", [](const mm::LimitLinear& ll) {
        return fmt::format(
            "LimitLinear(ref={}, target={}, scale={}, offset={})",
            ll.referenceIndex,
            ll.targetIndex,
            ll.scale,
            ll.offset);
      });

  parameterLimitLinearJointClass
      .def_readonly(
          "reference_joint_index",
          &mm::LimitLinearJoint::referenceJointIndex,
          "Index of reference joint to use in equation p_0 = scale * p_1 - offset.")
      .def_readonly(
          "reference_joint_parameter",
          &mm::LimitLinearJoint::referenceJointParameter,
          "Index of reference parameter to use (tx=0,ty=1,tz=2,rx=3,ry=4,rz=5,s=6).")
      .def_readonly(
          "target_joint_index",
          &mm::LimitLinearJoint::targetJointIndex,
          "Index of target joint to use in equation p_0 = scale * p_1 - offset.")
      .def_readonly(
          "target_joint_parameter",
          &mm::LimitLinearJoint::targetJointParameter,
          "Index of target parameter to use (tx=0,ty=1,tz=2,rx=3,ry=4,rz=5,s=6).")
      .def_readonly(
          "scale",
          &mm::LimitLinearJoint::scale,
          "Scale to use in equation p_0 = scale * p_1 - offset.")
      .def_readonly(
          "offset",
          &mm::LimitLinearJoint::offset,
          "Offset to use in equation p_0 = scale * p_1 - offset.")
      .def_property_readonly(
          "range_min",
          [](const mm::LimitLinearJoint& data) -> std::optional<float> {
            if (data.rangeMin <= -std::numeric_limits<float>::max()) {
              return std::optional<float>{};
            } else {
              return std::make_optional(data.rangeMin);
            }
          })
      .def_property_readonly(
          "range_max",
          [](const mm::LimitLinearJoint& data) -> std::optional<float> {
            if (data.rangeMax >= std::numeric_limits<float>::max()) {
              return std::optional<float>{};
            } else {
              return std::make_optional(data.rangeMax);
            }
          })
      .def("__repr__", [](const mm::LimitLinearJoint& llj) {
        return fmt::format(
            "LimitLinearJoint(ref_joint={}, ref_param={}, target_joint={}, target_param={}, scale={}, offset={})",
            llj.referenceJointIndex,
            llj.referenceJointParameter,
            llj.targetJointIndex,
            llj.targetJointParameter,
            llj.scale,
            llj.offset);
      });

  parameterLimitHalfPlaneClass.def_readonly("param1_index", &mm::LimitHalfPlane::param1)
      .def_readonly("param2_index", &mm::LimitHalfPlane::param2)
      .def_readonly("offset", &mm::LimitHalfPlane::offset)
      .def_readonly("normal", &mm::LimitHalfPlane::normal)
      .def("__repr__", [](const mm::LimitHalfPlane& lhp) {
        return fmt::format(
            "LimitHalfPlane(param1={}, param2={}, normal=[{} {}], offset={})",
            lhp.param1,
            lhp.param2,
            lhp.normal[0],
            lhp.normal[1],
            lhp.offset);
      });

  parameterLimitEllipsoidClass
      .def_property_readonly(
          "ellipsoid",
          [](const mm::LimitEllipsoid& data) -> Eigen::Matrix4f { return data.ellipsoid.matrix(); })
      .def_property_readonly(
          "ellipsoid_inv",
          [](const mm::LimitEllipsoid& data) -> Eigen::Matrix4f {
            return data.ellipsoidInv.matrix();
          })
      .def_readonly("offset", &mm::LimitEllipsoid::offset)
      .def_readonly("ellipsoid_parent", &mm::LimitEllipsoid::ellipsoidParent)
      .def_readonly("parent", &mm::LimitEllipsoid::parent)
      .def("__repr__", [](const mm::LimitEllipsoid& le) {
        return fmt::format(
            "LimitEllipsoid(ellipsoid_parent={}, parent={}, offset=[{} {} {}])",
            le.ellipsoidParent,
            le.parent,
            le.offset.x(),
            le.offset.y(),
            le.offset.z());
      });
}

} // namespace pymomentum
