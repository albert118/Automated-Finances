*Before you make a pull request please make sure you have ticked off the following things:*

Please, go through these steps before you submit a PR.

1. Make sure that your PR is not a duplicate.
2. If not, then make sure that:

    a. You have done your changes in a separate branch. Branches MUST have descriptive names that start with either the `fix/`, `hotfix/` or `feature/` prefixes. Good examples are: `fix/signin-issue` or `feature/issue-templates`.

    b. You have a descriptive commit message with a short title (first line).

    c. You have only one commit (if not, squash them into one commit).

    d. `npm test` doesn't throw any error. If it does, fix them first and amend your commit (`git commit --amend`).
    
    e. Local build runs without errors on visual studio on clean build.
    
    f. For major changes, build is cleaned, branch is rebased and solution rebuilt.

3. **After** these steps, you're ready to open a pull request.

    a. Your pull request MUST NOT target the `master` branch on this repository. You probably want to target `staging` instead.

    b. Give a descriptive title to your PR.

    c. Provide a description of your changes.

    d. Put `closes #XXXX` in your comment to auto-close the issue that your PR fixes (if such).

IMPORTANT: Please review the [CONTRIBUTING.md](../CONTRIBUTING.md) file for detailed contributing guidelines.

**PLEASE REMOVE ALL LINES ABOVE THIS TEMPLATE BEFORE SUBMITTING**

# Checklist
<!--- Check off each of these reminders! Put an `x` in all the boxes that apply: -->
- [ ] Tested UI changes for Webkit and Android (mobile)
- [ ] Tested Legacy UI changes for IE 11.
- [ ] Link a related issue (for fixes)
- [ ] Completed successful NPM build.
- [ ] Completed successful visual studio build.
- [ ] I have read the **CONTRIBUTING** document.
- [ ] Update the documentation accordingly.
- [ ] Add unit tests.

# Types of Changes
<!--- What types of changes does your code introduce? Put an `x` in all the boxes that apply: -->
- [ ] Bug fix (non-breaking change which fixes an issue).
- [ ] New feature (non-breaking change which adds functionality).
- [ ] Hotfix (quick data level duct-tape fix).
- [ ] Breaking change (fix or feature that would cause existing functionality to change).
- [ ] Major doc's update (You felt productive todat didn't you?).

# Data Testing Sources List
