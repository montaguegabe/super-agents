#!/usr/bin/env node
import { readFile } from "node:fs/promises";
import { join } from "node:path";

const MIN_AGE_DAYS = 10;
const root = process.cwd();

if (process.env.SUPER_AGENTS_SKIP_AGE_CHECK === "1") {
  process.exit(0);
}

const cutoff = Date.now() - MIN_AGE_DAYS * 24 * 60 * 60 * 1000;
const packages = await packageVersions();
const tooNew = [];
const unknown = [];

for (const [name, versions] of packages) {
  const metadata = await fetchPackageMetadata(name);
  const times = metadata.time ?? {};
  for (const version of versions) {
    const publishedAt = times[version];
    if (!publishedAt) {
      unknown.push(`${name}@${version}`);
      continue;
    }
    const publishedMs = Date.parse(publishedAt);
    if (Number.isFinite(publishedMs) && publishedMs > cutoff) {
      tooNew.push(`${name}@${version} published ${publishedAt}`);
    }
  }
}

if (unknown.length > 0) {
  console.error(`Could not verify package publication dates:\n${unknown.map((item) => `- ${item}`).join("\n")}`);
  process.exit(1);
}

if (tooNew.length > 0) {
  console.error(
    `Dependency age check failed. Packages must be at least ${MIN_AGE_DAYS} days old:\n` +
      tooNew.map((item) => `- ${item}`).join("\n"),
  );
  process.exit(1);
}

async function packageVersions() {
  const lock = JSON.parse(await readFile(join(root, "package-lock.json"), "utf8"));
  const entries = new Map();

  for (const [path, info] of Object.entries(lock.packages ?? {})) {
    if (!path || !path.startsWith("node_modules/") || !info || typeof info !== "object") {
      continue;
    }
    const name = packageNameFromLockPath(path);
    const version = info.version;
    if (!name || typeof version !== "string") {
      continue;
    }
    const versions = entries.get(name) ?? new Set();
    versions.add(version);
    entries.set(name, versions);
  }

  return entries;
}

async function fetchPackageMetadata(name) {
  const encodedName = encodeURIComponent(name);
  const response = await fetch(`https://registry.npmjs.org/${encodedName}`);
  if (!response.ok) {
    throw new Error(`Could not fetch npm metadata for ${name}: ${response.status}`);
  }
  return response.json();
}

function packageNameFromLockPath(path) {
  const parts = path.split("/").slice(1);
  const nodeModulesIndex = parts.lastIndexOf("node_modules");
  const packageParts = nodeModulesIndex >= 0 ? parts.slice(nodeModulesIndex + 1) : parts;
  if (packageParts[0]?.startsWith("@")) {
    return packageParts.length >= 2 ? `${packageParts[0]}/${packageParts[1]}` : undefined;
  }
  return packageParts[0];
}

