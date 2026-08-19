/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* eslint-env node */

const { spawnSync } = require("node:child_process");
const {
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} = require("node:fs");
const { tmpdir } = require("node:os");
const { dirname, join, resolve } = require("node:path");
const { pathToFileURL } = require("node:url");
const { deflateSync } = require("node:zlib");

const CHILD_PROCESS_TIMEOUT_MILLISECONDS = 5000;
const CLI_FILE_COUNT = 101;
const DOCS_IMAGE_DIRECTORIES = ["docs_cpp", "docs_python"];
const CLI_TIMEOUT_MILLISECONDS = 30000;
const LARGE_PNG_TRAILING_BYTES = 2 * 1024 * 1024;

const uint32Bytes = (value) => [
  (value >>> 24) & 0xff,
  (value >>> 16) & 0xff,
  (value >>> 8) & 0xff,
  value & 0xff,
];
const asciiBytes = (value) => [...Buffer.from(value, "ascii")];

function box(type, payload, extendsToEnd = false) {
  return [
    ...uint32Bytes(extendsToEnd ? 0 : 8 + payload.length),
    ...asciiBytes(type),
    ...payload,
  ];
}

function heifWithZeroLengthIspe() {
  return [
    ...box("ftyp", [...asciiBytes("mif1"), 0, 0, 0, 0]),
    ...box("meta", [
      0,
      0,
      0,
      0,
      ...box("iprp", [
        ...box("ipco", [
          ...uint32Bytes(0),
          ...asciiBytes("ispe"),
          0,
          0,
          0,
          0,
          ...uint32Bytes(32),
          ...uint32Bytes(16),
        ]),
      ]),
    ]),
  ];
}

function heifWithShortIspe() {
  return [
    ...box("ftyp", [...asciiBytes("mif1"), 0, 0, 0, 0]),
    ...box("meta", [
      0,
      0,
      0,
      0,
      ...box("iprp", [
        ...box("ipco", [
          ...uint32Bytes(1),
          ...asciiBytes("ispe"),
          0,
          0,
          0,
          0,
          ...uint32Bytes(32),
          ...uint32Bytes(16),
        ]),
      ]),
    ]),
  ];
}

function crc32(input) {
  let crc = 0xffffffff;
  for (const byte of input) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) {
      crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function pngChunk(type, data) {
  const typeBytes = Buffer.from(type, "ascii");
  const payload = Buffer.from(data);
  const crcInput = Buffer.concat([typeBytes, payload]);
  return Buffer.concat([
    Buffer.from(uint32Bytes(payload.length)),
    crcInput,
    Buffer.from(uint32Bytes(crc32(crcInput))),
  ]);
}

function ihdrData(width, height) {
  return Buffer.from([
    ...uint32Bytes(width),
    ...uint32Bytes(height),
    8,
    6,
    0,
    0,
    0,
  ]);
}

function pngPayload({ cgbi = false, height = 16, width = 32 } = {}) {
  const rowSize = width * 4 + 1;
  const imageData = Buffer.alloc(rowSize * height);
  const chunks = [];
  if (cgbi) {
    chunks.push(pngChunk("CgBI", Buffer.alloc(4)));
  }
  chunks.push(
    pngChunk("IHDR", ihdrData(width, height)),
    pngChunk("IDAT", deflateSync(imageData)),
    pngChunk("IEND", Buffer.alloc(0)),
  );
  return Buffer.concat([
    Buffer.from([0x89, ...asciiBytes("PNG\r\n\u001a\n")]),
    ...chunks,
  ]);
}

function invalidPng(mutate, { updateCrc = true } = {}) {
  const bytes = Buffer.from(pngPayload());
  mutate(bytes);
  if (updateCrc) {
    bytes.writeUInt32BE(crc32(bytes.subarray(12, 29)), 29);
  }
  return bytes;
}

const payloads = {
  cgbiPng: {
    bytes: pngPayload({ cgbi: true, height: 24, width: 48 }),
    expectedSize: { width: 48, height: 24, type: "png" },
  },
  invalidCgbiCrc: {
    bytes: (() => {
      const bytes = Buffer.from(pngPayload({ cgbi: true }));
      bytes[23] ^= 1;
      return bytes;
    })(),
    expectedError: /^Invalid PNG$/,
  },
  invalidPngCompression: {
    bytes: invalidPng((bytes) => {
      bytes[26] = 1;
    }),
    expectedError: /^Invalid PNG$/,
  },
  invalidPngCrc: {
    bytes: invalidPng(
      (bytes) => {
        bytes[32] ^= 1;
      },
      { updateCrc: false },
    ),
    expectedError: /^Invalid PNG$/,
  },
  invalidPngChunkLength: {
    bytes: invalidPng((bytes) => {
      bytes.writeUInt32BE(12, 8);
    }),
    expectedError: /^Invalid PNG$/,
  },
  invalidPngHeight: {
    bytes: invalidPng((bytes) => {
      bytes.writeUInt32BE(0, 20);
    }),
    expectedError: /^Invalid PNG$/,
  },
  invalidPngTruncated: {
    bytes: pngPayload().subarray(0, 24),
    expectedError: /^Invalid PNG$/,
  },
  invalidPngWidth: {
    bytes: invalidPng((bytes) => {
      bytes.writeUInt32BE(0x80000000, 16);
    }),
    expectedError: /^Invalid PNG$/,
  },
  jpeg: {
    bytes: [0xff, 0xd8, 0xff, 0xd9],
    expectedError: /^unsupported file type: undefined$/,
  },
  maliciousIcns: {
    bytes: [
      ...asciiBytes("icns"),
      ...uint32Bytes(16),
      ...asciiBytes("is32"),
      ...uint32Bytes(0),
    ],
    expectedError: /^unsupported file type: undefined$/,
  },
  maliciousIcnsShort: {
    bytes: [
      ...asciiBytes("icns"),
      ...uint32Bytes(16),
      ...asciiBytes("is32"),
      ...uint32Bytes(1),
    ],
    expectedError: /^unsupported file type: undefined$/,
  },
  maliciousHeif: {
    bytes: heifWithZeroLengthIspe(),
    expectedError: /^unsupported file type: undefined$/,
  },
  maliciousHeifShort: {
    bytes: heifWithShortIspe(),
    expectedError: /^unsupported file type: undefined$/,
  },
  maliciousJxl: {
    bytes: [
      ...box("JXL ", [0x0d, 0x0a, 0x87, 0x0a]),
      ...box("ftyp", [...asciiBytes("jxl "), 0, 0, 0, 0]),
      ...box("jxlp", [], true),
    ],
    expectedError: /^unsupported file type: undefined$/,
  },
  maliciousJxlShort: {
    bytes: [
      ...box("JXL ", [0x0d, 0x0a, 0x87, 0x0a]),
      ...box("ftyp", [...asciiBytes("jxl "), 0, 0, 0, 0]),
      0,
      0,
      0,
      1,
      ...asciiBytes("jxlp"),
      0,
      0,
      0,
      0,
    ],
    expectedError: /^unsupported file type: undefined$/,
  },
  png: {
    bytes: pngPayload(),
    expectedSize: { width: 32, height: 16, type: "png" },
  },
};

const childMode = process.argv[2];
const childFile = process.argv[3];

async function expectRejection(label, parse, expectedError) {
  try {
    await parse();
  } catch (error) {
    const message = error?.message ?? String(error);
    if (expectedError.test(message)) {
      return;
    }
    throw new Error(`${label} rejected with unexpected error: ${message}`);
  }
  throw new Error(`${label} accepted unsupported input`);
}

async function expectDimensions(label, parse, expectedSize) {
  const actualSize = await parse();
  for (const [key, value] of Object.entries(expectedSize)) {
    if (actualSize[key] !== value) {
      throw new Error(
        `${label} returned ${key}=${actualSize[key]}, expected ${value}`,
      );
    }
  }
}

async function runPayload(name, filePath) {
  if (!Object.prototype.hasOwnProperty.call(payloads, name)) {
    throw new Error(`Unknown image-size regression payload: ${name}`);
  }
  const payload = payloads[name];
  const { imageSize } = require("image-size");
  const { imageSizeFromFile } = require("image-size/fromFile");
  const [rootImport, fromFileImport] = await Promise.all([
    import("image-size"),
    import("image-size/fromFile"),
  ]);
  const checks = [
    [`require fromFile(${name})`, () => imageSizeFromFile(filePath)],
    [`require root(${name})`, () => imageSize(new Uint8Array(payload.bytes))],
    [`import fromFile(${name})`, () =>
      fromFileImport.imageSizeFromFile(filePath)],
    [`import root(${name})`, () =>
      rootImport.imageSize(new Uint8Array(payload.bytes))],
  ];

  for (const [label, parse] of checks) {
    if (payload.expectedSize) {
      await expectDimensions(label, parse, payload.expectedSize);
    } else {
      await expectRejection(label, parse, payload.expectedError);
    }
  }
}

async function runSpecialFile(filePath) {
  const { imageSizeFromFile } = require("image-size/fromFile");
  const fromFileImport = await import("image-size/fromFile");
  const packageDirectory = imageSizePackageDirectory();
  const absoluteFromFile = require(join(packageDirectory, "dist/fromFile.cjs"));
  const absoluteFromFileImport = await import(
    pathToFileURL(join(packageDirectory, "dist/fromFile.mjs")).href
  );
  const expectedError = /^Expected image path to reference a regular file$/;
  await expectRejection(
    "require fromFile(fifo)",
    () => imageSizeFromFile(filePath),
    expectedError,
  );
  await expectRejection(
    "import fromFile(fifo)",
    () => fromFileImport.imageSizeFromFile(filePath),
    expectedError,
  );
  await expectRejection(
    "absolute CJS fromFile(fifo)",
    () => absoluteFromFile.imageSizeFromFile(filePath),
    expectedError,
  );
  await expectRejection(
    "absolute ESM fromFile(fifo)",
    () => absoluteFromFileImport.imageSizeFromFile(filePath),
    expectedError,
  );
}

function expectEqual(label, actual, expected) {
  if (actual !== expected) {
    throw new Error(`${label}=${actual}, expected ${expected}`);
  }
}

function expectNotExported(label, load) {
  try {
    load();
  } catch (error) {
    if (error?.code === "ERR_PACKAGE_PATH_NOT_EXPORTED") {
      return;
    }
    throw new Error(`${label} failed with unexpected error: ${error.message}`);
  }
  throw new Error(`${label} unexpectedly resolved`);
}

function imageSizePackageDirectory() {
  return dirname(dirname(require.resolve("image-size")));
}

function markdownFiles(directory) {
  const files = [];
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) {
      files.push(...markdownFiles(path));
    } else if (entry.isFile() && /\.(?:md|mdx)$/.test(entry.name)) {
      files.push(path);
    }
  }
  return files;
}

function markdownReferenceLabel(label) {
  return label.trim().replace(/\s+/g, " ").toLowerCase();
}

function markdownDestination(rawDestination) {
  const trimmed = rawDestination.trim();
  if (trimmed.startsWith("<") && trimmed.endsWith(">")) {
    return trimmed.slice(1, -1);
  }
  return trimmed;
}

function expectPngMarkdownImage(filePath, target) {
  const normalizedTarget = markdownDestination(target).replace(/[?#].*$/, "");
  if (/^(?:https?:|data:|pathname:\/\/)/.test(normalizedTarget)) {
    return;
  }
  if (!normalizedTarget.toLowerCase().endsWith(".png")) {
    throw new Error(`${filePath} uses a non-PNG Markdown image: ${target}`);
  }
}

function expectMarkdownImagesArePng() {
  const definition = /^\s{0,3}\[([^\]]+)]:\s*(<[^>]+>|[^\s]+)(?:\s|$)/gm;
  const fullReferenceImage = /!\[([^\]]+)]\[([^\]]*)]/g;
  const inlineImage = /!\[[^\]]*]\(\s*(<[^>]+>|[^)\s]+)(?:\s+["'][^)]*)?\)/g;
  const shortcutReferenceImage = /!\[([^\]]+)](?![\[(])/g;
  for (const directory of DOCS_IMAGE_DIRECTORIES) {
    if (!statSync(directory).isDirectory()) {
      throw new Error(`${directory} is not a docs directory`);
    }
    for (const filePath of markdownFiles(directory)) {
      const contents = readFileSync(filePath, "utf8");
      const definitions = new Map();
      for (const match of contents.matchAll(definition)) {
        definitions.set(markdownReferenceLabel(match[1]), match[2]);
      }
      for (const match of contents.matchAll(inlineImage)) {
        expectPngMarkdownImage(filePath, match[1]);
      }
      for (const match of contents.matchAll(fullReferenceImage)) {
        const label = markdownReferenceLabel(match[2] || match[1]);
        if (!definitions.has(label)) {
          continue;
        }
        expectPngMarkdownImage(filePath, definitions.get(label));
      }
      for (const match of contents.matchAll(shortcutReferenceImage)) {
        const label = markdownReferenceLabel(match[1]);
        if (!definitions.has(label)) {
          continue;
        }
        expectPngMarkdownImage(filePath, definitions.get(label));
      }
    }
  }
}

function expectPackageSurface() {
  const packageJson = JSON.parse(
    readFileSync(join(imageSizePackageDirectory(), "package.json"), "utf8"),
  );
  const expectedEntries = [
    ["main", packageJson.main, "./dist/momentum-website-root-shim.cjs"],
    ["module", packageJson.module, "./dist/momentum-website-root-shim.mjs"],
    ["types", packageJson.types, "./dist/momentum-website-root-shim.d.ts"],
    [
      "root require export",
      packageJson.exports["."].require.default,
      "./dist/momentum-website-root-shim.cjs",
    ],
    [
      "root require types",
      packageJson.exports["."].require.types,
      "./dist/momentum-website-root-shim.d.ts",
    ],
    [
      "root import export",
      packageJson.exports["."].import.default,
      "./dist/momentum-website-root-shim.mjs",
    ],
    [
      "root import types",
      packageJson.exports["."].import.types,
      "./dist/momentum-website-root-shim.d.ts",
    ],
    [
      "fromFile require export",
      packageJson.exports["./fromFile"].require.default,
      "./dist/momentum-website-from-file-shim.cjs",
    ],
    [
      "fromFile require types",
      packageJson.exports["./fromFile"].require.types,
      "./dist/momentum-website-from-file-shim.d.ts",
    ],
    [
      "fromFile import export",
      packageJson.exports["./fromFile"].import.default,
      "./dist/momentum-website-from-file-shim.mjs",
    ],
    [
      "fromFile import types",
      packageJson.exports["./fromFile"].import.types,
      "./dist/momentum-website-from-file-shim.d.ts",
    ],
  ];
  for (const [label, actual, expected] of expectedEntries) {
    expectEqual(`image-size ${label}`, actual, expected);
  }
  if (Object.prototype.hasOwnProperty.call(packageJson.exports, "./types/*")) {
    throw new Error("image-size still exports parser internals");
  }
}

function expectShimResourceGuards() {
  const shimSource = readFileSync(
    join(imageSizePackageDirectory(), "dist/momentum-website-shim.cjs"),
    "utf8",
  );
  const expectations = [
    ["bounded header size", "const PNG_HEADER_SIZE = 49"],
    ["bounded file read", "Math.min(size, PNG_HEADER_SIZE)"],
    ["helper timeout", "timeout: FILE_READ_TIMEOUT_MILLISECONDS"],
    ["helper kill signal", 'killSignal: "SIGKILL"'],
    ["helper concurrency cap", "const MAX_CONCURRENCY = 4"],
    ["active helper limit", "activeJobs < concurrency"],
    ["nonblocking open", "O_NONBLOCK"],
    ["regular-file check", "stats.isFile()"],
  ];
  for (const [label, needle] of expectations) {
    if (!shimSource.includes(needle)) {
      throw new Error(`image-size shim missing ${label}`);
    }
  }
}

async function expectInternalParserGuards() {
  const packageDirectory = imageSizePackageDirectory();
  const internalRoot = require(join(packageDirectory, "dist/index.cjs"));
  const internalRootImport = await import(
    pathToFileURL(join(packageDirectory, "dist/index.mjs")).href
  );
  const internalLookup = require(join(packageDirectory, "dist/lookup.cjs"));
  const internalLookupImport = await import(
    pathToFileURL(join(packageDirectory, "dist/lookup.mjs")).href
  );
  const internalTypeIndex = require(
    join(packageDirectory, "dist/types/index.cjs"),
  );
  const internalTypeIndexImport = await import(
    pathToFileURL(join(packageDirectory, "dist/types/index.mjs")).href
  );
  const cases = [
    {
      cjsPath: "dist/types/heif.cjs",
      error: /^Invalid HEIF, invalid ispe box size$/,
      exportName: "HEIF",
      mjsPath: "dist/types/heif.mjs",
      name: "maliciousHeif",
      type: "heif",
    },
    {
      cjsPath: "dist/types/heif.cjs",
      error: /^Invalid HEIF, invalid ispe box size$/,
      exportName: "HEIF",
      mjsPath: "dist/types/heif.mjs",
      name: "maliciousHeifShort",
      type: "heif",
    },
    {
      cjsPath: "dist/types/icns.cjs",
      error: /^Invalid ICNS, invalid entry length$/,
      exportName: "ICNS",
      mjsPath: "dist/types/icns.mjs",
      name: "maliciousIcns",
      type: "icns",
    },
    {
      cjsPath: "dist/types/icns.cjs",
      error: /^Invalid ICNS, invalid entry length$/,
      exportName: "ICNS",
      mjsPath: "dist/types/icns.mjs",
      name: "maliciousIcnsShort",
      type: "icns",
    },
    {
      cjsPath: "dist/types/jxl.cjs",
      error: /^Invalid JXL, invalid jxlp box size$/,
      exportName: "JXL",
      mjsPath: "dist/types/jxl.mjs",
      name: "maliciousJxl",
      type: "jxl",
    },
    {
      cjsPath: "dist/types/jxl.cjs",
      error: /^Invalid JXL, invalid jxlp box size$/,
      exportName: "JXL",
      mjsPath: "dist/types/jxl.mjs",
      name: "maliciousJxlShort",
      type: "jxl",
    },
  ];
  for (const entry of cases) {
    const payload = new Uint8Array(payloads[entry.name].bytes);
    const cjsParser = require(join(packageDirectory, entry.cjsPath))[
      entry.exportName
    ];
    const mjsParser = await import(
      pathToFileURL(join(packageDirectory, entry.mjsPath)).href
    );
    await expectRejection(
      `internal root ${entry.name}`,
      () => internalRoot.imageSize(payload),
      entry.error,
    );
    await expectRejection(
      `internal ESM root ${entry.name}`,
      () => internalRootImport.imageSize(payload),
      entry.error,
    );
    await expectRejection(
      `internal CJS lookup ${entry.name}`,
      () => internalLookup.imageSize(payload),
      entry.error,
    );
    await expectRejection(
      `internal ESM lookup ${entry.name}`,
      () => internalLookupImport.imageSize(payload),
      entry.error,
    );
    await expectRejection(
      `internal CJS type index ${entry.name}`,
      () => internalTypeIndex.typeHandlers.get(entry.type).calculate(payload),
      entry.error,
    );
    await expectRejection(
      `internal ESM type index ${entry.name}`,
      () =>
        internalTypeIndexImport.typeHandlers
          .get(entry.type)
          .calculate(payload),
      entry.error,
    );
    await expectRejection(
      `internal CJS ${entry.name}`,
      () => cjsParser.calculate(payload),
      entry.error,
    );
    await expectRejection(
      `internal ESM ${entry.name}`,
      () => mjsParser[entry.exportName].calculate(payload),
      entry.error,
    );
  }
}

async function expectRootApi(pngFilePath) {
  const root = require("image-size");
  const fromFile = require("image-size/fromFile");
  const rootImport = await import("image-size");
  const fromFileImport = await import("image-size/fromFile");

  expectEqual("image-size __esModule", root.__esModule, true);
  expectEqual("image-size types", root.types.join(","), "png");
  expectEqual("image-size ESM types", rootImport.types.join(","), "png");
  expectEqual("image-size CJS default type", typeof root.default, "function");
  expectEqual("image-size ESM default type", typeof rootImport.default, "function");
  expectEqual(
    "image-size CJS imageSizeFromFile type",
    typeof root.imageSizeFromFile,
    "undefined",
  );
  expectEqual(
    "image-size ESM imageSizeFromFile type",
    typeof rootImport.imageSizeFromFile,
    "undefined",
  );
  expectEqual(
    "image-size CJS setConcurrency type",
    typeof root.setConcurrency,
    "undefined",
  );
  expectEqual(
    "image-size ESM setConcurrency type",
    typeof rootImport.setConcurrency,
    "undefined",
  );
  expectEqual(
    "image-size/fromFile CJS default type",
    typeof fromFile.default,
    "undefined",
  );
  expectEqual(
    "image-size/fromFile CJS disableTypes type",
    typeof fromFile.disableTypes,
    "undefined",
  );
  expectEqual(
    "image-size/fromFile ESM disableTypes type",
    typeof fromFileImport.disableTypes,
    "undefined",
  );
  await expectDimensions(
    "image-size CJS default",
    () => root.default(new Uint8Array(payloads.png.bytes)),
    payloads.png.expectedSize,
  );
  await expectDimensions(
    "image-size ESM default",
    () => rootImport.default(new Uint8Array(payloads.png.bytes)),
    payloads.png.expectedSize,
  );

  const disabled = ["png"];
  try {
    root.disableTypes(disabled);
    disabled.length = 0;
    await expectRejection(
      "imageSize(disabled png)",
      () => root.imageSize(new Uint8Array(payloads.png.bytes)),
      /^disabled file type: png$/,
    );
    await expectDimensions(
      "imageSizeFromFile(disabled ignored)",
      () => fromFile.imageSizeFromFile(pngFilePath),
      payloads.png.expectedSize,
    );
    await expectDimensions(
      "import imageSizeFromFile(disabled ignored)",
      () => fromFileImport.imageSizeFromFile(pngFilePath),
      payloads.png.expectedSize,
    );
  } finally {
    root.disableTypes([]);
  }
}

async function expectLargePngFile(temporaryDirectory) {
  const filePath = join(temporaryDirectory, "large.png");
  const bytes = Buffer.concat([
    Buffer.from(payloads.png.bytes),
    Buffer.alloc(LARGE_PNG_TRAILING_BYTES),
  ]);
  writeFileSync(filePath, bytes);

  const { imageSize } = require("image-size");
  const { imageSizeFromFile, setConcurrency } = require("image-size/fromFile");
  await expectDimensions(
    "imageSize(large png)",
    () =>
      imageSize(
        new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength),
      ),
    payloads.png.expectedSize,
  );

  setConcurrency(1);
  await Promise.all(
    Array.from({ length: 4 }, (_, index) =>
      expectDimensions(
        `imageSizeFromFile(large png ${index})`,
        () => imageSizeFromFile(filePath),
        payloads.png.expectedSize,
      ),
    ),
  );
  setConcurrency(Number.MAX_SAFE_INTEGER);
  await expectDimensions(
    "imageSizeFromFile(capped concurrency)",
    () => imageSizeFromFile(filePath),
    payloads.png.expectedSize,
  );
}

function expectCliUsesShim(pngFilePath, maliciousJxlPath) {
  const result = spawnSync(
    process.execPath,
    [
      join(imageSizePackageDirectory(), "bin/image-size.js"),
      ...Array.from({ length: CLI_FILE_COUNT - 1 }, () => pngFilePath),
      maliciousJxlPath,
    ],
    { encoding: "utf8", timeout: CLI_TIMEOUT_MILLISECONDS },
  );
  assertChildCompleted(result, "image-size CLI");
  const plainStdout = result.stdout.replace(/\x1B\[[0-?]*[ -/]*[@-~]/g, "");
  const resultCount = plainStdout.split("32x16").length - 1;
  const plainStderr = result.stderr.replace(/\x1B\[[0-?]*[ -/]*[@-~]/g, "");
  if (
    result.status !== 0 ||
    resultCount !== CLI_FILE_COUNT - 1 ||
    !plainStderr.includes("unsupported file type: undefined") ||
    !plainStderr.includes(maliciousJxlPath)
  ) {
    const details =
      result.stderr || result.stdout || `exit status ${result.status}`;
    throw new Error(`image-size CLI did not use the shim: ${details}`);
  }
}

async function expectFilesystemError(temporaryDirectory) {
  const missingPath = join(temporaryDirectory, "missing.png");
  const { imageSizeFromFile } = require("image-size/fromFile");
  const fromFileImport = await import("image-size/fromFile");
  for (const [label, parse] of [
    ["require fromFile(missing)", () => imageSizeFromFile(missingPath)],
    [
      "import fromFile(missing)",
      () => fromFileImport.imageSizeFromFile(missingPath),
    ],
  ]) {
    try {
      await parse();
    } catch (error) {
      expectEqual(`${label} code`, error.code, "ENOENT");
      expectEqual(`${label} syscall`, error.syscall, "open");
      expectEqual(`${label} path`, error.path, resolve(missingPath));
      continue;
    }
    throw new Error(`${label} unexpectedly succeeded`);
  }
}

async function expectAbsoluteFromFileApi(pngFilePath, temporaryDirectory) {
  const missingPath = join(temporaryDirectory, "absolute-missing.png");
  const packageDirectory = imageSizePackageDirectory();
  const absoluteFromFile = require(join(packageDirectory, "dist/fromFile.cjs"));
  const absoluteFromFileImport = await import(
    pathToFileURL(join(packageDirectory, "dist/fromFile.mjs")).href
  );
  for (const [label, api] of [
    ["absolute CJS fromFile", absoluteFromFile],
    ["absolute ESM fromFile", absoluteFromFileImport],
  ]) {
    expectEqual(`${label} default type`, typeof api.default, "undefined");
    await expectDimensions(
      `${label}(png)`,
      () => api.imageSizeFromFile(pngFilePath),
      payloads.png.expectedSize,
    );
    try {
      await api.imageSizeFromFile(missingPath);
    } catch (error) {
      expectEqual(`${label}(missing) code`, error.code, "ENOENT");
      expectEqual(`${label}(missing) syscall`, error.syscall, "open");
      expectEqual(`${label}(missing) path`, error.path, resolve(missingPath));
      continue;
    }
    throw new Error(`${label}(missing) unexpectedly succeeded`);
  }
}

function expectBoundedHelperRead(temporaryDirectory) {
  const packageDirectory = imageSizePackageDirectory();
  const preloadPath = join(temporaryDirectory, "bounded-read-preload.cjs");
  writeFileSync(
    preloadPath,
    `
const fsPromises = require("node:fs/promises");
const header = Uint8Array.from(${JSON.stringify(
      [...payloads.png.bytes].slice(0, 49),
    )});
fsPromises.open = async () => ({
  stat: async () => ({ isFile: () => true, size: 1024 * 1024 * 1024 }),
  read: async (buffer, offset, length, position) => {
    if (length > 49) {
      throw new Error(\`read length exceeded PNG header: \${length}\`);
    }
    const chunk = header.subarray(position, position + length);
    buffer.set(chunk, offset);
    return { bytesRead: chunk.length };
  },
  close: async () => {},
});
`,
  );
  const result = spawnSync(
    process.execPath,
    [
      "--require",
      preloadPath,
      join(packageDirectory, "dist/momentum-website-shim.cjs"),
      "--read-file",
      "instrumented.png",
      "0",
    ],
    { encoding: "utf8", timeout: CHILD_PROCESS_TIMEOUT_MILLISECONDS },
  );
  assertChildCompleted(result, "bounded helper read");
  const parsed = JSON.parse(result.stdout);
  if (parsed.error) {
    throw new Error(`bounded helper read failed: ${parsed.error.message}`);
  }
  expectEqual("bounded helper read width", parsed.size.width, 32);
  expectEqual("bounded helper read height", parsed.size.height, 16);
}

async function expectConcurrencyCap() {
  const childProcess = require("node:child_process");
  let active = 0;
  let peak = 0;
  childProcess.execFile = (_file, _args, _options, callback) => {
    active += 1;
    peak = Math.max(peak, active);
    setTimeout(() => {
      active -= 1;
      callback(null, JSON.stringify({ size: payloads.png.expectedSize }), "");
    }, 20);
    return {};
  };

  const { imageSizeFromFile, setConcurrency } = require("image-size/fromFile");
  setConcurrency(Number.MAX_SAFE_INTEGER);
  await Promise.all(
    Array.from({ length: 8 }, () => imageSizeFromFile("instrumented.png")),
  );
  expectEqual("imageSizeFromFile peak capped concurrency", peak, 4);

  peak = 0;
  setConcurrency(1);
  await Promise.all(
    Array.from({ length: 3 }, () => imageSizeFromFile("instrumented.png")),
  );
  expectEqual("imageSizeFromFile peak serial concurrency", peak, 1);
}

function assertChildCompleted(result, label) {
  if (result.error?.code === "ETIMEDOUT") {
    throw new Error(`${label} hung`);
  }
  if (result.error) {
    throw new Error(`${label} could not run: ${result.error.message}`);
  }
  if (result.signal !== null) {
    throw new Error(`${label} was terminated by ${result.signal}`);
  }
  if (result.status !== 0) {
    const details =
      result.stderr || result.stdout || `exit status ${result.status}`;
    throw new Error(`${label} failed: ${details}`);
  }
}

function runChild(mode, filePath) {
  const args = [__filename, mode];
  if (filePath !== undefined) {
    args.push(filePath);
  }
  const result = spawnSync(process.execPath, args, {
    encoding: "utf8",
    timeout: CHILD_PROCESS_TIMEOUT_MILLISECONDS,
  });
  assertChildCompleted(result, `image-size ${mode} regression`);
}

async function expectSpecialFileRejection(temporaryDirectory) {
  if (process.platform === "win32") {
    return;
  }
  const fifoPath = join(temporaryDirectory, "image.fifo");
  const result = spawnSync("mkfifo", [fifoPath], {
    encoding: "utf8",
    timeout: CHILD_PROCESS_TIMEOUT_MILLISECONDS,
  });
  assertChildCompleted(result, "mkfifo");
  runChild("fifo", fifoPath);
}

async function runRegressionChecks() {
  const temporaryDirectory = mkdtempSync(
    join(tmpdir(), "image-size-security-"),
  );
  try {
    for (const [name, payload] of Object.entries(payloads)) {
      const filePath = join(temporaryDirectory, name);
      writeFileSync(filePath, new Uint8Array(payload.bytes));
      runChild(name, filePath);
    }

    expectNotExported("image-size/types/heif", () =>
      require("image-size/types/heif"),
    );
    await expectRejection(
      "import(image-size/types/heif)",
      () => import("image-size/types/heif"),
      /Package subpath '.\/types\/heif' is not defined by "exports"/,
    );
    expectMarkdownImagesArePng();
    expectPackageSurface();
    expectShimResourceGuards();
    runChild("internal-guards");
    runChild("concurrency");
    expectBoundedHelperRead(temporaryDirectory);
    await expectRootApi(join(temporaryDirectory, "png"));
    await expectLargePngFile(temporaryDirectory);
    await expectSpecialFileRejection(temporaryDirectory);
    await expectFilesystemError(temporaryDirectory);
    await expectAbsoluteFromFileApi(
      join(temporaryDirectory, "png"),
      temporaryDirectory,
    );
    expectCliUsesShim(
      join(temporaryDirectory, "png"),
      join(temporaryDirectory, "maliciousJxl"),
    );
  } finally {
    rmSync(temporaryDirectory, { force: true, recursive: true });
  }
}

if (childMode) {
  const run =
    childMode === "fifo"
      ? () => runSpecialFile(childFile)
      : childMode === "internal-guards"
        ? () => expectInternalParserGuards()
        : childMode === "concurrency"
          ? () => expectConcurrencyCap()
      : () => runPayload(childMode, childFile);
  run().catch((error) => {
    console.error(error);
    process.exit(1);
  });
} else {
  runRegressionChecks()
    .then(() => {
      console.log("image-size security regression checks passed");
    })
    .catch((error) => {
      console.error(error);
      process.exit(1);
    });
}
