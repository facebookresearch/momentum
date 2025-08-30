---
sidebar_position: 2
---

# Style Guide

<FbInternalOnly>
Momentum adheres to the following upstream standards, except when explicitly mentioned to override specific items:

- [Meta's C++ Coding Conventions](https://www.internalfb.com/intern/wiki/Cpp/CppCodingConventions/)
- [Meta's C++ Style Guide](https://www.internalfb.com/intern/wiki/Cpp/CppStyle/)
- [Common Pitfalls](https://www.internalfb.com/intern/wiki/C++_Common_Pitfalls_and_Best_Practices/)
</FbInternalOnly>

<OssOnly>
Momentum adheres to specific coding conventions and style guidelines that are based on internal standards developed by Meta. While these specific documents are not publicly accessible, we encourage external developers to follow general best practices in C++ programming to ensure code quality and maintainability.
</OssOnly>

## Error Handling

Momentum uses exceptions for handling unrecoverable errors, which is essential for a lower-level library like this. Throwing exceptions allows error detection mechanisms to have visibility into these errors and prevents silent failures, which may lead to more severe bugs. By choosing exceptions as the primary method for error handling in Momentum, we ensure that unrecoverable errors are easily detectable, and the Momentum library remains user-friendly for developers interacting with the application layer.

Using exceptions is especially crucial when considering the application layer as the user of the Momentum library. For layers closer to the services, everything should be surrounded by try-catch blocks to prevent server crashes while still providing valuable error information. Python programmers, who often interact with the application layer, typically expect exceptions, making it a more reasonable approach for error handling in Momentum.

One important caution is to avoid using exceptions for flow control; common errors, such as "L2 norm is too high," should not result in exceptions being thrown.

### Alternative Approaches

#### std::options

While optional types can be acceptable in some cases, they can lead to the loss of error information, which is not ideal for the application layer. We use optional types for inputs that can be missing for a few frames due to reasons such as lost tracking, but in these cases, the specific reason is not critical.

#### folly::Expected

Folly::Expected acts like an optional type but allows for an error type to be specified (see documentation [here](https://www.internalfb.com/code/fbsource/[84b294fbc7bb1d3a90efcdd440a9d7a4d9f83222]/xplat/folly/Expected.h?lines=818)). This can be useful in some APIs, particularly if error codes need to be serialized or for other similar purposes. In general, folly::Expected is preferred over std::optional.

### Array Access

When accessing elements within arrays, the Momentum codebase employs two methods: the `.at()` method and the `[]` operator. The choice between these two methods should be made based on the considerations of performance, safety, and the intended audience of the code:

- It is recommended to use the `[]` operator for code that is self-contained and developed by the Momentum team, especially for low-level and performance-critical code. The `[]` operator does not perform bounds checking, thus offering better performance.

- Conversely, the `.at()` method should be considered for areas of code that might be accessed by the user as it performs bounds checking to prevent out-of-range errors, enhancing the safety of the code.

## Design Decisions

### Prefer `gsl::span` over typed container like `std::vector` as function argument

Using `std::vector<T>` as a function argument requires the call site to create an `std::vector<T>` instance even when the data is already stored in a compatible memory layout, such as contiguous memory. By switching to `gsl::span<T>`, call sites can avoid creating an additional `std::vector<T>` object and benefit from improved performance by not requiring an unnecessary data copy.

### Header Include Style

Momentum follows standard C++ conventions for header includes, using different syntax depending on the context:

#### Public Headers (Use Angle Brackets `<>`)

In **public Momentum headers** that are installed for users, use angle brackets:

```cpp
// In public header files (e.g., momentum/character/character.h)
#include <momentum/common/checks.h>
#include <momentum/math/mesh.h>
```

**Rationale**: Angle brackets instruct the compiler to search in system/standard include paths. When Momentum is installed as a library, downstream users and build systems will treat these as system headers and look for them in the appropriate system locations (e.g., `/usr/local/include`, `/opt/include`).

#### Library Implementation Source Files (Use Quotes `""`)

In **library implementation source files** (`.cpp` files in core library directories), use quotes:

```cpp
// In library implementation .cpp files
#include "momentum/character/character.h"
#include "momentum/common/checks.h"
```

**Rationale**: Quotes instruct the compiler to first search in local/relative directories before falling back to system paths. This ensures that during Momentum's own build process, the compiler finds the local development headers rather than any previously installed system versions.

#### Tests, Tutorials, and Examples (Use Angle Brackets `<>`)

In **tests, tutorials, and example code**, use angle brackets:

```cpp
// In test files, tutorials, and examples
#include <momentum/character/character.h>
#include <momentum/common/checks.h>
```

**Rationale**: Tests, tutorials, and examples represent user-facing code that demonstrates how external developers should consume the Momentum library. They should mirror the external user experience by using angle brackets as users would when the library is installed as a system library. This also serves as documentation showing the correct way to include headers.
