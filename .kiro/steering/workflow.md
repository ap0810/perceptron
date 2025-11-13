## Workflow Rules

### TODO Document Requirements

**STRICT ENFORCEMENT:** All TODO documents must follow these rules:

1. **Signature Required**
   - TODO.md MUST have a signature section at the bottom
   - TODO.md MUST include a warning at the top: "⚠️ **WARNING:** This document must be signed before pushing to GitHub."
   - Never push TODO.md to GitHub without a valid signature and date

2. **Session Logging**
   - Every work session MUST be logged in TODO.md
   - Include date and time (or time of day: morning/afternoon/evening)
   - Document what was completed during the session
   - Document what needs to be done next

3. **Format Requirements**
   ```markdown
   # TODO - Future Work
   
   > ⚠️ **WARNING:** This document must be signed before pushing to GitHub.
   
   **Last Updated:** [Date and Time]
   
   [TODO items...]
   
   ## Session Notes - [Date] ([Time Period])
   
   ### Completed Today
   [List of completed items]
   
   ### Next Session
   [List of next items]
   
   ---
   
   **Signed:** _[Signature Required]_  
   **Date:** _[Date Required]_
   ```

4. **Pre-Push Checklist**
   - Before any `git push` command, verify TODO.md is either:
     - Signed and ready to push, OR
     - Added to .gitignore temporarily if not ready
   - Remind the user to sign TODO.md if attempting to push unsigned version

5. **Session End Protocol**
   - When user indicates end of session (e.g., "taking a break", "done for today")
   - Automatically update TODO.md with session notes
   - Remind user that signature is required before next push
