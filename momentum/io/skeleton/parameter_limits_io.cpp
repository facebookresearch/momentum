/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/skeleton/parameter_limits_io.h"

#include <limits>

#include "momentum/character/parameter_transform.h"
#include "momentum/character/skeleton.h"
#include "momentum/common/log.h"
#include "momentum/math/constants.h"
#include "momentum/math/utility.h"

namespace momentum {

namespace {

const float kNegativeInfinity = -std::numeric_limits<float>::max();
const float kPositiveInfinity = std::numeric_limits<float>::max();

enum class Token {
  OpenBracket,
  CloseBracket,
  Number,
  Comma,
  Identifier,
  Eof,
};

std::string tokenStr(const Token& t) {
  switch (t) {
    case Token::OpenBracket:
      return "[";
    case Token::CloseBracket:
      return "]";
    case Token::Number:
      return "float value";
    case Token::Comma:
      return ",";
    case Token::Eof:
      return "EOF";
    case Token::Identifier:
      return "Identifier";
  }
  return "Unknown";
}

class Tokenizer {
 public:
  Tokenizer(
      const std::string& str,
      size_t lineIndex,
      const ParameterTransform& parameterTransform,
      const Skeleton& skeleton)
      : parameterTransform(parameterTransform),
        skeleton(skeleton),
        _str(str),
        _lineIndex(lineIndex) {
    _curPos = _str.begin();
    _tokenStart = _curPos;
    advance();
  }

  [[nodiscard]] bool isEOF() const {
    return _curToken == Token::Eof;
  }

  void eatToken(Token expectedToken) {
    verifyToken(expectedToken);
    advance();
  }

  void expectToken(Token expectedToken) {
    verifyToken(expectedToken);
  }

  [[nodiscard]] Token peek() const {
    return _curToken;
  }

  [[nodiscard]] std::string getIdentifier() {
    verifyToken(Token::Identifier);
    std::string result(_tokenStart, _curPos);
    advance();
    return result;
  }

  [[nodiscard]] float getNumber() {
    verifyToken(Token::Number);
    float result = std::stof(std::string(_tokenStart, _curPos));
    advance();
    return result;
  }

  template <int N>
  [[nodiscard]] Vector<float, N> getVec() {
    Vector<float, N> result;

    eatToken(Token::OpenBracket);
    for (Eigen::Index i = 0; i < N; ++i) {
      if (i > 0) {
        eatToken(Token::Comma);
      }
      result[i] = getNumber();
    }
    eatToken(Token::CloseBracket);

    return result;
  }

  // Get a vector with an arbitrary number of elements.
  [[nodiscard]] VectorXf getDynamicVec() {
    std::vector<float> result;

    eatToken(Token::OpenBracket);
    while (peek() != Token::CloseBracket) {
      if (!result.empty()) {
        eatToken(Token::Comma);
      }
      result.push_back(getNumber());
    }
    eatToken(Token::CloseBracket);

    Eigen::VectorXf vec(result.size());
    for (Eigen::Index i = 0; i < vec.size(); ++i) {
      vec(i) = result.at(i);
    }
    return vec;
  }

  [[nodiscard]] std::pair<float, float> getPair() {
    Vector2<float> result = getVec<2>();
    return {result.x(), result.y()};
  }

  [[nodiscard]] size_t modelParameterIndexFromName(const std::string& modelParameterName) const {
    auto itr = std::find(
        parameterTransform.name.begin(), parameterTransform.name.end(), modelParameterName);
    MT_THROW_IF(
        itr == parameterTransform.name.end(),
        "Parameter {} not found in transform at line {}: {}",
        modelParameterName,
        _lineIndex,
        _str);
    return std::distance(parameterTransform.name.begin(), itr);
  }

  [[nodiscard]] size_t getModelParameterIndex() {
    return modelParameterIndexFromName(getIdentifier());
  }

  [[nodiscard]] std::pair<size_t, size_t> jointParameterIndexFromName(
      const std::string& jointParameterName) const {
    auto dotPos = jointParameterName.find('.');
    MT_THROW_IF(
        dotPos == std::string::npos,
        "Could not find '.' in joint parameter name {}; expected 'joint_name.tx/tz/tz/rx/ry/rz/s' at line {}: {}",
        jointParameterName,
        _lineIndex,
        _str);

    auto jointName = jointParameterName.substr(0, dotPos);
    const auto jointIndex = jointIndexFromName(jointName);

    const auto subParamName = jointParameterName.substr(dotPos + 1);
    const auto* kJointParameterNamesEnd = kJointParameterNames.data() + kParametersPerJoint;
    const auto* subParamItr =
        std::find_if(kJointParameterNames.data(), kJointParameterNamesEnd, [&](const char* name) {
          return subParamName == name;
        });
    MT_THROW_IF(
        subParamItr == kJointParameterNamesEnd,
        "Invalid parameter name {}, expected tx/ty/tz/rx/ry/rz/s at line {}: {}",
        subParamName,
        _lineIndex,
        _str);
    return {jointIndex, std::distance(kJointParameterNames.data(), subParamItr)};
  }

  [[nodiscard]] std::pair<size_t, size_t> getJointParameterIndex() {
    return jointParameterIndexFromName(getIdentifier());
  }

  [[nodiscard]] size_t jointIndexFromName(const std::string& jointName) const {
    const auto jointIndex = skeleton.getJointIdByName(jointName);
    MT_THROW_IF(
        jointIndex == kInvalidIndex,
        "Could not find joint {} in skeleton at line {}: {}",
        jointName,
        _lineIndex,
        _str);

    return jointIndex;
  }

  [[nodiscard]] size_t getJointIndex() {
    return jointIndexFromName(getIdentifier());
  }

  [[nodiscard]] size_t currentPosition() const {
    return std::distance(_str.begin(), _curPos);
  }

  [[nodiscard]] size_t lineIndex() const {
    return _lineIndex;
  }

  [[nodiscard]] const std::string& str() const {
    return _str;
  }

  const ParameterTransform& parameterTransform;
  const Skeleton& skeleton;

 private:
  void verifyToken(Token expectedToken) const {
    MT_THROW_IF(
        _curToken != expectedToken,
        "Expected {} at character {} of line {}: {}",
        tokenStr(expectedToken),
        std::distance(_str.begin(), _tokenStart),
        _lineIndex,
        _str);
  }

  void advance() {
    // Skip whitespace
    while (_curPos != _str.end() && std::isspace(*_curPos)) {
      ++_curPos;
    }

    _tokenStart = _curPos;
    if (_curPos == _str.end()) {
      _curToken = Token::Eof;
      return;
    }

    if (*_curPos == '[') {
      _curToken = Token::OpenBracket;
      ++_curPos;
      return;
    }

    if (*_curPos == ']') {
      _curToken = Token::CloseBracket;
      ++_curPos;
      return;
    }

    if (*_curPos == ',') {
      _curToken = Token::Comma;
      ++_curPos;
      return;
    }

    // We won't allow numbers to start with a period so you'll need to say 0.1 instead of .1,
    // this will help to avoid ambiguous cases whre the dot is part of a joint parameter definition.
    if (std::isdigit(*_curPos) || *_curPos == '-' || *_curPos == '+') {
      _curToken = Token::Number;
      while (_curPos != _str.end() &&
             (std::isdigit(*_curPos) || *_curPos == '-' || *_curPos == '+' || *_curPos == '.' ||
              *_curPos == 'e')) {
        ++_curPos;
      }
      return;
    }

    if (_curPos != _str.end() && (std::isalpha(*_curPos) || *_curPos == '_')) {
      _curToken = Token::Identifier;
      // period in identifier is okay, we consider "joint.rx" as a single identifier
      while (std::isalnum(*_curPos) || *_curPos == '_' || *_curPos == '.') {
        ++_curPos;
      }
      return;
    }

    MT_THROW(
        "Unexpected token at character {} of line {}: {}",
        std::distance(_str.begin(), _tokenStart),
        _lineIndex,
        _str);
  }

  const std::string& _str;
  const size_t _lineIndex;
  std::string::const_iterator _curPos;
  std::string::const_iterator _tokenStart;
  Token _curToken = Token::Eof;
};

void parseMinmax(const std::string& parameterName, ParameterLimits& pl, Tokenizer& tokenizer) {
  if (parameterName.find('.') == std::string::npos) {
    // "[<min>, <max>] <optional weight>"
    ParameterLimit p;
    p.weight = 1.0f;
    p.type = MinMax;
    p.data.minMax.parameterIndex = tokenizer.modelParameterIndexFromName(parameterName);

    std::tie(p.data.minMax.limits.x(), p.data.minMax.limits.y()) = tokenizer.getPair();
    if (!tokenizer.isEOF()) {
      p.weight = tokenizer.getNumber();
    }
    pl.push_back(p);
  } else {
    // "[<min>, <max>] <optional weight>"
    ParameterLimit p;
    p.weight = 1.0f;
    p.type = MinMaxJoint;
    std::tie(p.data.minMaxJoint.jointIndex, p.data.minMaxJoint.jointParameter) =
        tokenizer.jointParameterIndexFromName(parameterName);

    std::tie(p.data.minMaxJoint.limits.x(), p.data.minMaxJoint.limits.y()) = tokenizer.getPair();
    if (!tokenizer.isEOF()) {
      p.weight = tokenizer.getNumber();
    }
    pl.push_back(p);
  }
}

void parseMinmaxPassive(
    const std::string& parameterName,
    ParameterLimits& pl,
    Tokenizer& tokenizer) {
  // "[<min>, <max>] <optional weight>"
  ParameterLimit p;
  p.weight = 1.0f;
  p.type = MinMaxJointPassive;
  std::tie(p.data.minMaxJoint.jointIndex, p.data.minMaxJoint.jointParameter) =
      tokenizer.jointParameterIndexFromName(parameterName);

  std::tie(p.data.minMaxJoint.limits.x(), p.data.minMaxJoint.limits.y()) = tokenizer.getPair();
  if (!tokenizer.isEOF()) {
    p.weight = tokenizer.getNumber();
  }
  pl.push_back(p);
}

void parseLinear(const std::string& parameterName, ParameterLimits& pl, Tokenizer& tokenizer) {
  // create new parameterlimit
  ParameterLimit p;
  p.weight = 1.0f;
  p.type = Linear;
  p.data.linear.referenceIndex = tokenizer.modelParameterIndexFromName(parameterName);

  // "<model parameter name> [<segment1_scale>, <segment1_offset>, segment1_rangeEnd>]
  // [<segment2_scale> <segment2_offset> <segment2_rangeEnd>] ... [<segment_n_scale>
  // <segment_n_offset] <optional weight>"
  //
  // Each segment of the piecewise-linear limit is specified
  // sequentially as [scale, offset, rangeEnd] where rangeEnd specifies the _upper_ limit where that
  // segment applies.
  //
  // Example:
  //  f(x) = { -x - 3        if x < -3
  //           x - (-3)         if -3 <= x < 0
  //           -2x - (-3)       if x >= 0
  // This would look like:
  //    limit param1 linear param2 [-1, 3, -3] [1, -3, 0] [-2, -3] 4.0
  // The missing end range for the last segment implies that it ranges to +infinity.
  p.data.linear.targetIndex = tokenizer.getModelParameterIndex();

  tokenizer.expectToken(Token::OpenBracket);

  const auto sizeBefore = pl.size();

  auto evalFunction = [](const LimitLinear& limit, float value) -> float {
    return limit.scale * value - limit.offset;
  };

  float prevRangeMax = kNegativeInfinity;
  while (tokenizer.peek() == Token::OpenBracket) {
    const auto linePos = tokenizer.currentPosition();
    const VectorXf segment = tokenizer.getDynamicVec();

    if (segment.size() < 2 || segment.size() > 3) {
      MT_THROW(
          "Expected 2 or 3 values for linear segment at position {}, got {} at line {}: {}",
          segment.size(),
          linePos,
          tokenizer.lineIndex(),
          tokenizer.str());
    }

    if (prevRangeMax == kPositiveInfinity && segment.size() == 3) {
      MT_THROW(
          "Only the last linear segment can have unrestricted range at position {} in line {}: {}",
          linePos,
          tokenizer.lineIndex(),
          tokenizer.str());
    }

    const float curRangeMax = segment.size() == 3 ? segment[2] : kPositiveInfinity;

    auto pCur = p;
    pCur.data.linear.scale = segment[0];
    pCur.data.linear.offset = segment[1];
    pCur.data.linear.rangeMin = prevRangeMax;
    pCur.data.linear.rangeMax = curRangeMax;

    MT_LOGT(
        "Parsed linear (joint) parameter limit {} = {}*{} - {}",
        tokenizer.parameterTransform.name.at(pCur.data.linear.referenceIndex),
        pCur.data.linearJoint.scale,
        tokenizer.parameterTransform.name.at(pCur.data.linear.targetIndex),
        pCur.data.linearJoint.offset);

    if (pl.size() > sizeBefore) {
      const auto& pPrev = pl.back();
      MT_CHECK(pPrev.type == pCur.type);
      const auto valuePrev = evalFunction(pPrev.data.linear, prevRangeMax);
      const auto valueCur = evalFunction(pCur.data.linear, prevRangeMax);

      MT_THROW_IF(
          std::abs(valuePrev - valueCur) > 1e-3f,
          "Mismatch between function values between two linear segments in line {}: {}.  "
          " At x={}, {}*{} - {} = {} vs {}*{} - {} = {}",
          tokenizer.lineIndex(),
          tokenizer.str(),
          prevRangeMax,
          pPrev.data.linear.scale,
          prevRangeMax,
          pPrev.data.linear.offset,
          valuePrev,
          pCur.data.linear.scale,
          prevRangeMax,
          pCur.data.linear.offset,
          valueCur);
    }
    pl.push_back(pCur);

    prevRangeMax = curRangeMax;
  }

  if (!tokenizer.isEOF()) {
    p.weight = tokenizer.getNumber();
  }

  // Back-fill all the weights:
  for (size_t i = sizeBefore; i < pl.size(); ++i) {
    pl[i].weight = p.weight;
  }
}

void parseLinearJoint(const std::string& parameterName, ParameterLimits& pl, Tokenizer& tokenizer) {
  // create new parameterlimit
  ParameterLimit p;
  p.weight = 1.0f;
  p.type = LinearJoint;
  std::tie(p.data.linearJoint.referenceJointIndex, p.data.linearJoint.referenceJointParameter) =
      tokenizer.jointParameterIndexFromName(parameterName);

  // "<model parameter name> [<segment1_scale>, <segment1_offset>, segment1_rangeEnd>]
  // [<segment2_scale> <segment2_offset> <segment2_rangeEnd>] ... [<segment_n_scale>
  // <segment_n_offset] <optional weight>"
  //
  // Each segment of the piecewise-linear limit is specified
  // sequentially as [scale, offset, rangeEnd] where rangeEnd specifies the _upper_ limit where that
  // segment applies.
  //
  // Example:
  //  f(x) = { -x - 3        if x < -3
  //           x - (-3)         if -3 <= x < 0
  //           -2x - (-3)       if x >= 0
  // This would look like:
  //    limit param1 linear param2 [-1, 3, -3] [1, -3, 0] [-2, -3] 4.0
  // The missing end range for the last segment implies that it ranges to +infinity.
  std::tie(p.data.linearJoint.targetJointIndex, p.data.linearJoint.targetJointParameter) =
      tokenizer.getJointParameterIndex();

  tokenizer.expectToken(Token::OpenBracket);

  const auto sizeBefore = pl.size();

  auto evalFunction = [](const LimitLinearJoint& limit, float value) -> float {
    return limit.scale * value - limit.offset;
  };

  float prevRangeMax = kNegativeInfinity;
  while (tokenizer.peek() == Token::OpenBracket) {
    const auto linePos = tokenizer.currentPosition();
    const VectorXf segment = tokenizer.getDynamicVec();

    if (segment.size() < 2 || segment.size() > 3) {
      MT_THROW(
          "Expected 2 or 3 values for linear segment at position {}, got {} at line {}: {}",
          segment.size(),
          linePos,
          tokenizer.lineIndex(),
          tokenizer.str());
    }

    if (prevRangeMax == kPositiveInfinity && segment.size() == 3) {
      MT_THROW(
          "Only the last linear segment can have unrestricted range at position {} in line {}: {}",
          linePos,
          tokenizer.lineIndex(),
          tokenizer.str());
    }

    const float curRangeMax = segment.size() == 3 ? segment[2] : kPositiveInfinity;

    auto pCur = p;
    pCur.data.linearJoint.scale = segment[0];
    pCur.data.linearJoint.offset = segment[1];
    pCur.data.linearJoint.rangeMin = prevRangeMax;
    pCur.data.linearJoint.rangeMax = curRangeMax;

    MT_LOGT(
        "Parsed linear (joint) parameter limit {}.{} = {}*{}.{} - {}",
        tokenizer.skeleton.joints.at(pCur.data.linearJoint.referenceJointIndex).name,
        kJointParameterNames[pCur.data.linearJoint.referenceJointParameter],
        pCur.data.linearJoint.scale,
        tokenizer.skeleton.joints.at(pCur.data.linearJoint.targetJointIndex).name,
        kJointParameterNames[pCur.data.linearJoint.targetJointParameter],
        pCur.data.linearJoint.offset);

    if (pl.size() > sizeBefore) {
      const auto& pPrev = pl.back();
      MT_CHECK(pPrev.type == pCur.type);
      const auto valuePrev = evalFunction(pPrev.data.linearJoint, prevRangeMax);
      const auto valueCur = evalFunction(pCur.data.linearJoint, prevRangeMax);

      MT_THROW_IF(
          std::abs(valuePrev - valueCur) > 1e-3f,
          "Mismatch between function values between two linear segments in line {}: {}.  "
          " At x={}, {}*{} - {} = {} vs {}*{} - {} = {}",
          tokenizer.lineIndex(),
          tokenizer.str(),
          prevRangeMax,
          pPrev.data.linearJoint.scale,
          prevRangeMax,
          pPrev.data.linearJoint.offset,
          valuePrev,
          pCur.data.linearJoint.scale,
          prevRangeMax,
          pCur.data.linearJoint.offset,
          valueCur);
    }
    pl.push_back(pCur);

    prevRangeMax = curRangeMax;
  }

  if (!tokenizer.isEOF()) {
    p.weight = tokenizer.getNumber();
  }

  // Back-fill all the weights:
  for (size_t i = sizeBefore; i < pl.size(); ++i) {
    pl[i].weight = p.weight;
  }
}

void parseHalfPlane(const std::string& parameterName, ParameterLimits& pl, Tokenizer& tokenizer) {
  ParameterLimit p;
  p.weight = 1.0f;
  p.type = HalfPlane;
  p.data.halfPlane.param1 = tokenizer.modelParameterIndexFromName(parameterName);
  p.data.halfPlane.param2 = tokenizer.getModelParameterIndex();
  p.data.halfPlane.normal = tokenizer.getVec<2>();
  p.data.halfPlane.offset = tokenizer.getNumber();

  // Normalize the constraint:
  const float len = p.data.halfPlane.normal.norm();
  p.data.halfPlane.normal /= len;
  p.data.halfPlane.offset /= len;

  if (!tokenizer.isEOF()) {
    p.weight = tokenizer.getNumber();
  }
  pl.push_back(p);
}

void parseEllipsoid(const std::string& jointName, ParameterLimits& pl, Tokenizer& tokenizer) {
  // create new parameterlimit
  ParameterLimit p;
  p.weight = 1.0f;
  p.type = Ellipsoid;

  // "[offset] parent [translation] [rotation] [scale] <optional weight>"
  p.data.ellipsoid.parent = tokenizer.jointIndexFromName(jointName);
  p.data.ellipsoid.offset = tokenizer.getVec<3>();

  p.data.ellipsoid.ellipsoidParent = tokenizer.getJointIndex();

  const Eigen::Vector3f translation = tokenizer.getVec<3>();
  const Eigen::Vector3f eulerZYX = tokenizer.getVec<3>();
  const Eigen::Vector3f scale = tokenizer.getVec<3>();

  if (!tokenizer.isEOF()) {
    p.weight = tokenizer.getNumber();
  }

  // parse ellipsoid number data
  p.data.ellipsoid.ellipsoid = Affine3f::Identity();
  p.data.ellipsoid.ellipsoid.translation() = translation;
  const Vector3f eulerXYZ = Vector3f(toRad(eulerZYX.z()), toRad(eulerZYX.y()), toRad(eulerZYX.x()));
  p.data.ellipsoid.ellipsoid.linear() =
      eulerXYZToRotationMatrix(eulerXYZ, EulerConvention::Extrinsic) *
      Eigen::Scaling(scale.x(), scale.y(), scale.z());
  p.data.ellipsoid.ellipsoidInv = p.data.ellipsoid.ellipsoid.inverse();

  pl.push_back(std::move(p));
}

} // namespace

ParameterLimits parseParameterLimits(
    const std::string& data,
    const Skeleton& skeleton,
    const ParameterTransform& parameterTransform) {
  ParameterLimits pl;

  std::istringstream f(data);
  std::string line;
  size_t lineIndex = 0;
  while (std::getline(f, line)) {
    ++lineIndex;

    // erase all comments
    line = line.substr(0, line.find_first_of('#'));

    // ignore everything but limits
    if (line.find("limit") != 0) {
      continue;
    }

    // parse limits
    // token 1 = parameter name, token 2 = type, token 3 = value
    Tokenizer tokenizer(line, lineIndex, parameterTransform, skeleton);

    // Eat the initial "limit":
    std::ignore = tokenizer.getIdentifier();

    const std::string parameterName = tokenizer.getIdentifier();

    const std::string type = tokenizer.getIdentifier();
    // create a new ParameterLimit by the type and add it to pl
    if (type == "minmax") {
      parseMinmax(parameterName, pl, tokenizer);
    } else if (type == "minmax_passive") {
      parseMinmaxPassive(parameterName, pl, tokenizer);
    } else if (type == "linear") {
      if (parameterName.find('.') == std::string::npos) {
        parseLinear(parameterName, pl, tokenizer);
      } else {
        parseLinearJoint(parameterName, pl, tokenizer);
      }
    } else if (type == "halfplane") {
      parseHalfPlane(parameterName, pl, tokenizer);
    } else if (type == "ellipsoid") {
      parseEllipsoid(parameterName, pl, tokenizer);
    } else if (type == "elipsoid") {
      MT_LOGW_ONCE(
          "Deprecated parameter limit type: {} (typo). Please use 'ellipsoid' instead.", type);
      parseEllipsoid(parameterName, pl, tokenizer);
    } else {
      MT_THROW("Unexpected parameter limit type {} in line {}: {}", type, lineIndex, line);
    }

    MT_THROW_IF(
        !tokenizer.isEOF(),
        "Unexpected token at character {} in line {}: {}",
        tokenizer.currentPosition(),
        lineIndex,
        line);
  }
  return pl;
}

namespace {

std::string vecToString(Eigen::Ref<const Eigen::VectorXf> vec) {
  std::ostringstream oss;
  oss << "[";
  for (Eigen::Index i = 0; i < vec.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << vec[i];
  }
  oss << "]";

  return oss.str();
}

} // namespace

std::string writeParameterLimits(
    const ParameterLimits& parameterLimits,
    const Skeleton& skeleton,
    const ParameterTransform& parameterTransform) {
  std::ostringstream oss;

  auto jointParameterToName = [&](size_t jointIndex, size_t jointParameterIndex) {
    MT_CHECK_LT(jointParameterIndex, kParametersPerJoint);
    MT_CHECK_LT(jointIndex, skeleton.joints.size());
    return (skeleton.joints.at(jointIndex).name + "." + kJointParameterNames[jointParameterIndex]);
  };

  auto itr = parameterLimits.begin();
  while (itr != parameterLimits.end()) {
    oss << "limit ";

    // Linear needs to be handled specially because it can have multiple segments.
    if (itr->type == LimitType::Linear) {
      oss << parameterTransform.name.at(itr->data.linear.referenceIndex) << " linear "
          << parameterTransform.name.at(itr->data.linear.targetIndex);

      const auto origLimit = *itr;
      float prevRangeMax = kNegativeInfinity;

      // The .data gets initialized with all zeros (using memset) so we need to
      // handle the case where both rangeMin and rangeMax are zero.
      if (itr->data.linear.rangeMin == 0.0f && itr->data.linear.rangeMax == 0.0f) {
        oss << " " << vecToString(Eigen::Vector2f(itr->data.linear.scale, itr->data.linear.offset));
        ++itr;
      } else {
        do {
          oss << " ";
          if (itr->data.linear.rangeMax == kPositiveInfinity) {
            oss << vecToString(Eigen::Vector2f(itr->data.linear.scale, itr->data.linear.offset));
          } else {
            oss << vecToString(
                Eigen::Vector3f(
                    itr->data.linear.scale, itr->data.linear.offset, itr->data.linear.rangeMax));
          }

          prevRangeMax = itr->data.linear.rangeMax;
          ++itr;
        } while (itr != parameterLimits.end() && itr->type == LimitType::Linear &&
                 itr->data.linear.referenceIndex == origLimit.data.linear.referenceIndex &&
                 itr->data.linear.targetIndex == origLimit.data.linear.targetIndex &&
                 itr->data.linear.rangeMin == prevRangeMax);
      }
      oss << " " << origLimit.weight << "\n";
      continue;
    } else if (itr->type == LimitType::LinearJoint) {
      oss << jointParameterToName(
                 itr->data.linearJoint.referenceJointIndex,
                 itr->data.linearJoint.referenceJointParameter)
          << " linear "
          << jointParameterToName(
                 itr->data.linearJoint.targetJointIndex,
                 itr->data.linearJoint.targetJointParameter);

      const auto origLimit = *itr;
      float prevRangeMax = kNegativeInfinity;

      // The .data gets initialized with all zeros (using memset) so we need to
      // handle the case where both rangeMin and rangeMax are zero.
      if (itr->data.linearJoint.rangeMin == 0.0f && itr->data.linearJoint.rangeMax == 0.0f) {
        oss << " "
            << vecToString(
                   Eigen::Vector2f(itr->data.linearJoint.scale, itr->data.linearJoint.offset));
        ++itr;
      } else {
        do {
          oss << " ";
          if (itr->data.linearJoint.rangeMax == kPositiveInfinity) {
            oss << vecToString(
                Eigen::Vector2f(itr->data.linearJoint.scale, itr->data.linearJoint.offset));
          } else {
            oss << vecToString(
                Eigen::Vector3f(
                    itr->data.linearJoint.scale,
                    itr->data.linearJoint.offset,
                    itr->data.linearJoint.rangeMax));
          }

          prevRangeMax = itr->data.linearJoint.rangeMax;
          ++itr;
        } while (itr != parameterLimits.end() && itr->type == LimitType::LinearJoint &&
                 itr->data.linearJoint.referenceJointIndex ==
                     origLimit.data.linearJoint.referenceJointIndex &&
                 itr->data.linearJoint.referenceJointParameter ==
                     origLimit.data.linearJoint.referenceJointParameter &&
                 itr->data.linearJoint.targetJointIndex ==
                     origLimit.data.linearJoint.targetJointIndex &&
                 itr->data.linearJoint.targetJointParameter ==
                     origLimit.data.linearJoint.targetJointParameter &&
                 itr->data.linearJoint.rangeMin == prevRangeMax);
      }
      oss << " " << origLimit.weight << "\n";
      continue;
    }

    switch (itr->type) {
      case LimitType::MinMax:
        oss << parameterTransform.name.at(itr->data.minMax.parameterIndex) << " minmax "
            << vecToString(itr->data.minMax.limits);
        break;
      case LimitType::MinMaxJoint:
        oss << jointParameterToName(
                   itr->data.minMaxJoint.jointIndex, itr->data.minMaxJoint.jointParameter)
            << " minmax " << vecToString(itr->data.minMaxJoint.limits);
        break;
      case LimitType::MinMaxJointPassive:
        oss << jointParameterToName(
                   itr->data.minMaxJoint.jointIndex, itr->data.minMaxJoint.jointParameter)
            << " minmax_passive " << vecToString(itr->data.minMaxJoint.limits);
        break;
      case LimitType::Linear: // Handled above
      case LimitType::LinearJoint: // Handled above
        break;
      case LimitType::HalfPlane:
        oss << parameterTransform.name.at(itr->data.halfPlane.param1) << " halfplane "
            << parameterTransform.name.at(itr->data.halfPlane.param2) << " "
            << vecToString(itr->data.halfPlane.normal) << " " << itr->data.halfPlane.offset;
        break;
      case LimitType::Ellipsoid: {
        const Eigen::Affine3f& ellipsoid = itr->data.ellipsoid.ellipsoid;
        const Eigen::Vector3f translation = ellipsoid.translation();
        Eigen::Matrix3f rotationMat;
        Eigen::Matrix3f scalingMat;
        ellipsoid.computeRotationScaling(&rotationMat, &scalingMat);

        const Eigen::Vector3f eulerXYZVecRad =
            rotationMatrixToEulerXYZ(rotationMat, EulerConvention::Extrinsic);
        // Not sure why this is reversed but it's implemented this way in the parser above.
        const Eigen::Vector3f eulerZYXVecDeg(
            toDeg(eulerXYZVecRad.z()), toDeg(eulerXYZVecRad.y()), toDeg(eulerXYZVecRad.x()));

        MT_CHECK_LT(itr->data.ellipsoid.parent, skeleton.joints.size());
        MT_CHECK_LT(itr->data.ellipsoid.ellipsoidParent, skeleton.joints.size());
        oss << skeleton.joints.at(itr->data.ellipsoid.parent).name << " ellipsoid "
            << vecToString(itr->data.ellipsoid.offset) << " "
            << skeleton.joints.at(itr->data.ellipsoid.ellipsoidParent).name << " "
            << vecToString(translation) << " " << vecToString(eulerZYXVecDeg) << " "
            << vecToString(scalingMat.diagonal());
        break;
      }
      case LimitType::LimitTypeCount:
        // do nothing
        break;
    }

    if (itr->weight != 1.0f) {
      oss << " " << itr->weight;
    }
    oss << "\n";
    ++itr;
  }

  return oss.str();
}

} // namespace momentum
