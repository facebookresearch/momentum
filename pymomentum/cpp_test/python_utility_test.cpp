/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/character.h>
#include <momentum/character/skin_weights.h>
#include <momentum/math/mesh.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pymomentum/python_utility/python_utility.h>

#include <gtest/gtest.h>

namespace py = pybind11;

// Python module for testing
PYBIND11_EMBEDDED_MODULE(test_module, m) {
  // Define the Character class
  py::class_<momentum::Character, std::shared_ptr<momentum::Character>>(
      m, "Character")
      .def(py::init<>())
      .def_readwrite(
          "parameterTransform", &momentum::Character::parameterTransform)
      .def(
          "set_mesh",
          [](momentum::Character& c, momentum::Mesh* mesh) {
            c.mesh.reset(mesh);
          })
      // Add a method to set the skinWeights pointer
      .def("set_skin_weights", [](momentum::Character& c, bool hasSkinWeights) {
        if (hasSkinWeights) {
          // Create a new SkinWeights object
          c.skinWeights = std::make_unique<momentum::SkinWeights>();
        } else {
          // Set the skinWeights pointer to null
          c.skinWeights.reset();
        }
      });

  // Define the ParameterTransform class
  py::class_<momentum::ParameterTransform>(m, "ParameterTransform")
      .def(py::init<>())
      .def_readwrite("name", &momentum::ParameterTransform::name);

  // Define the Mesh class (not as a unique_ptr)
  py::class_<momentum::Mesh>(m, "Mesh")
      .def(py::init<>())
      .def_readwrite("vertices", &momentum::Mesh::vertices)
      .def("add_vertex", [](momentum::Mesh& mesh, float x, float y, float z) {
        mesh.vertices.push_back(Eigen::Vector3f(x, y, z));
      });
}

// Global Python interpreter guard to ensure Python is initialized once
// and finalized at the end of the program
static py::scoped_interpreter guard{};

// Test fixture for Python utility tests
class PythonUtilityTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Python interpreter is already initialized by the global guard

    // Import our test module
    test_module = py::module::import("test_module");
  }

  void TearDown() override {
    // Python interpreter will be finalized by the global guard at program exit
  }

  // Store shared_ptrs to keep characters alive during the test
  std::vector<std::shared_ptr<momentum::Character>> characters_;

  // Python module for testing
  py::module test_module;
};

// Test from_msgpack and to_msgpack
TEST_F(PythonUtilityTest, MsgpackConversion) {
  // Create a JSON object
  nlohmann::json json = {
      {"name", "test"}, {"value", 42}, {"nested", {{"key", "value"}}}};

  // Convert to msgpack
  py::bytes msgpack = pymomentum::to_msgpack(json);

  // Convert back to JSON
  nlohmann::json result = pymomentum::from_msgpack(msgpack);

  // Verify the result
  EXPECT_EQ(result["name"], "test");
  EXPECT_EQ(result["value"], 42);
  EXPECT_EQ(result["nested"]["key"], "value");
}

// Test PyBytesStreamBuffer
TEST_F(PythonUtilityTest, BytesStreamBuffer) {
  // Create a Python bytes object
  std::string data = "Hello, world!";
  py::bytes pyBytes(data);

  // Create a stream buffer
  pymomentum::PyBytesStreamBuffer buffer(pyBytes);

  // Create an input stream using the buffer
  std::istream stream(&buffer);

  // Read from the stream
  std::string result;
  std::getline(stream, result);

  // Verify the result
  EXPECT_EQ(result, "Hello, world!");
}

// Test from_msgpack with invalid data
TEST_F(PythonUtilityTest, FromMsgpackInvalid) {
  // Create invalid msgpack data
  py::bytes invalidBytes("not valid msgpack data");

  // Test with invalid data
  EXPECT_THROW(
      pymomentum::from_msgpack(invalidBytes), nlohmann::json::parse_error);
}

// Test PyBytesStreamBuffer with empty bytes
TEST_F(PythonUtilityTest, BytesStreamBufferEmpty) {
  // Create an empty Python bytes object
  py::bytes emptyBytes("");

  // Create a stream buffer
  pymomentum::PyBytesStreamBuffer buffer(emptyBytes);

  // Create an input stream using the buffer
  std::istream stream(&buffer);

  // Try to read from the stream
  std::string result;
  std::getline(stream, result);

  // Verify the result is empty
  EXPECT_TRUE(result.empty());
  EXPECT_TRUE(stream.eof());
}

// Tests using the actual Python utility functions
class PythonUtilityPythonTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Python interpreter is already initialized by the global guard

    // Import our test module
    test_module = py::module::import("test_module");
  }

  void TearDown() override {
    // Python interpreter will be finalized by the global guard at program exit
  }

  // Helper method to create a Python Character object
  py::object createPyCharacter(
      const std::string& name,
      int numParams = 10,
      bool hasMesh = true,
      int numVertices = 100,
      bool hasSkinWeights = false) {
    // Create a Character object using our test module
    py::object character = test_module.attr("Character")();

    // Create a ParameterTransform object
    py::object paramTransform = test_module.attr("ParameterTransform")();

    // Set the name
    py::list nameList;
    nameList.append(name);
    paramTransform.attr("name") = nameList;

    // Set the parameter transform
    character.attr("parameterTransform") = paramTransform;

    // Set up mesh if requested
    if (hasMesh) {
      py::object mesh = test_module.attr("Mesh")();

      // Add vertices using the add_vertex method
      for (int i = 0; i < numVertices; ++i) {
        mesh.attr("add_vertex")(0.0f, 0.0f, static_cast<float>(i));
      }

      // Use the set_mesh method instead of directly setting the mesh property
      character.attr("set_mesh")(mesh);
    }

    // Set up skin weights if requested
    if (hasSkinWeights) {
      // Use our custom method to set the skinWeights pointer
      character.attr("set_skin_weights")(true);
    }

    return character;
  }

  // Python module for testing
  py::module test_module;

  // Store Python objects to keep them alive during the test
  std::vector<py::object> py_objects;
};

// Test anyCharacter with empty list
TEST_F(PythonUtilityPythonTest, PythonAnyCharacterEmptyList) {
  // Create an empty Python list
  py::list emptyList;

  // Test anyCharacter with empty list
  EXPECT_THROW(
      pymomentum::anyCharacter(emptyList.ptr(), "test"), std::runtime_error);
}

// Test anyCharacter with invalid object
TEST_F(PythonUtilityPythonTest, PythonAnyCharacterInvalidObject) {
  // Create a Python object that is not a Character
  py::object notACharacter = py::int_(42);

  // Test anyCharacter with invalid object
  EXPECT_THROW(
      pymomentum::anyCharacter(notACharacter.ptr(), "test"),
      std::runtime_error);
}

// Test anyCharacter with a list containing an invalid object
TEST_F(PythonUtilityPythonTest, PythonAnyCharacterListWithInvalidObject) {
  // Create a Python list with an invalid object
  py::list mixedList;
  mixedList.append(py::int_(42)); // Not a Character object

  // Test anyCharacter with a list containing an invalid object
  EXPECT_THROW(
      pymomentum::anyCharacter(mixedList.ptr(), "test"), std::runtime_error);
}

// Tests using the actual Python utility functions with our Python module
TEST_F(PythonUtilityPythonTest, PythonToCharacterListSingle) {
  // Create a Python Character object
  py::object character = createPyCharacter("test_character");

  // Test with forceBatchSize = true
  auto result = pymomentum::toCharacterList(character.ptr(), 3, "test", true);
  ASSERT_EQ(result.size(), 3);
  for (const auto& c : result) {
    ASSERT_FALSE(c->parameterTransform.name.empty());
    EXPECT_EQ(c->parameterTransform.name[0], "test_character");
  }

  // Test with forceBatchSize = false
  result = pymomentum::toCharacterList(character.ptr(), 3, "test", false);
  ASSERT_EQ(result.size(), 1);
  ASSERT_FALSE(result[0]->parameterTransform.name.empty());
  EXPECT_EQ(result[0]->parameterTransform.name[0], "test_character");
}

TEST_F(PythonUtilityPythonTest, PythonToCharacterListMultiple) {
  // Create a list of Python Character objects
  py::list characterList;
  characterList.append(createPyCharacter("test_character", 10, true, 100));
  characterList.append(createPyCharacter("test_character", 10, true, 100));
  characterList.append(createPyCharacter("test_character", 10, true, 100));

  // Test with correct batch size
  auto result = pymomentum::toCharacterList(characterList.ptr(), 3, "test");
  ASSERT_EQ(result.size(), 3);
  for (const auto& c : result) {
    ASSERT_FALSE(c->parameterTransform.name.empty());
    EXPECT_EQ(c->parameterTransform.name[0], "test_character");
  }
}

// Test toCharacterList with mismatched skin weights
TEST_F(PythonUtilityPythonTest, PythonToCharacterListMismatchedSkinWeights) {
  // Create a list of Python Character objects with different skin weights
  py::list characterList;
  characterList.append(
      createPyCharacter("test_character", 10, true, 100, true));
  characterList.append(
      createPyCharacter("test_character", 10, true, 100, false));

  // Test with mismatched skin weights
  EXPECT_THROW(
      pymomentum::toCharacterList(characterList.ptr(), 2, "test"),
      std::runtime_error);
}

// Test toCharacterList with invalid single object
TEST_F(PythonUtilityPythonTest, PythonToCharacterListInvalidSingle) {
  // Create a Python object that is not a Character
  py::object notACharacter = py::int_(42);

  // Store Python object to keep it alive during the test
  py_objects.push_back(notACharacter);

  // Test toCharacterList with invalid object
  EXPECT_THROW(
      pymomentum::toCharacterList(notACharacter.ptr(), 1, "test"),
      std::runtime_error);
}

// Test toCharacterList with a list containing an invalid object
TEST_F(PythonUtilityPythonTest, PythonToCharacterListInvalidInList) {
  // Create a list with a single invalid object
  py::list invalidList;
  py::object notACharacter = py::int_(42);

  // Store Python objects to keep them alive during the test
  py_objects.push_back(notACharacter);

  invalidList.append(notACharacter);

  // Store the list to keep it alive during the test
  py_objects.push_back(invalidList);

  // Test toCharacterList with a list containing an invalid object
  EXPECT_THROW(
      pymomentum::toCharacterList(invalidList.ptr(), 1, "test"),
      std::runtime_error);
}
