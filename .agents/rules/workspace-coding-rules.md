---
trigger: always_on
---

# Workspace AI Coding Rules

## 1. Understand This Project First

* Treat the current workspace as the source of truth for the project.
* Inspect the project structure before making architectural decisions.
* Read relevant source files, configuration files, and documentation before modifying them.
* Search the workspace for existing implementations before creating new ones.
* Understand how the affected feature currently works before changing it.

## 2. Follow the Existing Architecture

* Follow the project's existing architecture, folder structure, and coding patterns.
* Reuse existing modules, components, services, utilities, and helpers when possible.
* Do not introduce a completely different architecture for a small feature.
* Do not change the overall architecture unless the task genuinely requires it.

## 3. Understand Dependencies Between Features

* Before changing a shared file, function, component, service, API, or data structure, check what depends on it.
* Consider whether the change can affect other existing features.
* Be especially careful with shared utilities, authentication, navigation, database models, API contracts, and global state.

## 4. Preserve Previous Fixes

* Treat features that were previously confirmed as working as a baseline.
* Do not break a previously working feature while fixing another feature.
* When changing code related to a previous fix, preserve the behavior that was already working.
* If a new change causes a previous feature to stop working, fix the regression before finishing the task.

## 5. Feature Development

* Identify the relevant files before implementing a feature.
* Make the smallest reasonable change that satisfies the request.
* Extend existing functionality when possible instead of replacing it.
* Preserve existing behavior unless the user explicitly requests a change.
* Keep new functionality consistent with the existing project.

## 6. UI Changes

* Inspect the existing UI before modifying it.
* Preserve the existing design, layout, navigation, and behavior unless a redesign is requested.
* Do not modify unrelated screens or components.
* Reuse existing UI components and styles when possible.
* Avoid changing working UI merely for personal preference.

## 7. Backend and Data

* Understand the existing data flow before modifying backend code.
* Preserve existing API contracts unless a change is specifically required.
* Do not change database schemas or data structures unnecessarily.
* Preserve existing validation, authentication, authorization, and error-handling behavior.
* Check affected frontend/backend interactions after API changes.

## 8. Dependencies and Configuration

* Prefer dependencies already installed in the project.
* Do not add libraries unless they are actually needed.
* Do not change dependency versions unnecessarily.
* Do not modify build, environment, or configuration files unless required.
* Explain important configuration changes.

## 9. Scope Control

* Do not perform unrelated cleanup or refactoring while completing a task.
* Do not modify files simply because they could be improved.
* Do not rename or reorganize existing code unless necessary.
* Keep changes focused on the current request.

## 10. Testing

* Test the feature or code path affected by the change.
* Check compilation, build, runtime, lint, or type errors where applicable.
* Verify important existing functionality that could have been affected.
* When a change affects shared code, perform additional checks on dependent functionality.
* If complete testing is not possible, clearly report what was and was not tested.

## 11. Review Before Finishing

Before declaring the task complete:

* Review the changes made.
* Check for accidental modifications.
* Check for obvious errors.
* Check that previous working functionality has not been unnecessarily changed.
* Confirm that the requested task was actually completed.
* Report any remaining issues or limitations.

## 12. Project-Specific Rules

* Follow any additional instructions documented in this workspace.
* Project-specific requirements take priority over generic assumptions about how the project should work.
* When project documentation conflicts with an assumption, inspect the actual implementation and follow the project's documented requirements.
