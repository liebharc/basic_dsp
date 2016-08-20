$all_okay = $true

$last_master_update = $(git show --format="%ci" "origin/master" | Select -First 1)
$last_master_update = $(Get-Date $last_master_update)

# Release needs to be tagged
$current_version = $(Get-Content Cargo.toml | Select-String -Pattern "version\s+=\s+""(\d+\.\d+\.\d+)""" -AllMatches | % { $_.Matches.Groups[1].Value })
$current_tag = "v$current_version"
$tag_matches =  $(git tag | Where { $_ -eq "$current_tag" } | Measure).Count
if ($tag_matches -eq 0) {
    $Host.UI.WriteErrorLine("Tag $current_tag is missing. Run ""git tag $current_tag; git push --tags""")
    $all_okay = $false
}
else {
    $tag_date = $(git show --format="%ci" "$current_tag" | Select -First 1);
    $tag_date = $(Get-Date $tag_date)
        $age_difference_days = ($last_master_update - $tag_date).TotalDays
    if ($age_difference_days -gt 3.0) {
        $Host.UI.WriteErrorLine("Tag $current_tag is $age_difference_days days older than the latest master commit, did you forget to update the lib version?")
        $all_okay = $false
    }
}
# Release needs to in the changelog
$changelog_matches =  $(Get-Content Changelog.md | Where { $_ -match "$current_version" } | Measure).Count
if ($changelog_matches -eq 0) {
    $Host.UI.WriteErrorLine("Version $current_version isn't mentioned in Changelog.md")
}

# Documentation needs to be up to date
$last_doc_update = $(git show --format="%ci" "origin/gh-pages" | Select -First 1)
$last_doc_update = $(Get-Date $last_doc_update)

$age_difference_days = ($last_master_update - $last_doc_update).TotalDays

if ($age_difference_days -gt 3.0) {
    $Host.UI.WriteErrorLine("Documentation hasn't been updated $age_difference_days days since the last master commit.")
    $all_okay = $false
}

# Interop facade32 (which is the "master") needs to be complete
$cwd = $(Get-Location)
cd src\interop_facade
$facade_status = $(perl facade32_check_completness.pl)
$missing_ops_in_facade = $(echo $facade_status | Select-String -Pattern "missing:\s*(\d+)" -AllMatches | % { $_.Matches.Groups[1].Value })
if ($missing_ops_in_facade -gt 0) {
    $Host.UI.WriteErrorLine("There seem to be $missing_ops_in_facade operations missing in interop facade.")
    $all_okay = $false
}

# Interop facade64 must be cloned from facade32
perl facade64_create.pl | Write-Verbose
$facade64_diff = $(git diff .\facade64.rs)
if ($facade64_diff -gt 0) {
    $Host.UI.WriteErrorLine("facade64.rs seems to miss updates. Please review the changes and check them in.")
    $all_okay = $false
}

cd $cwd

if ($all_okay) {
    Write-Output "Everything seems to be okay, run ""cargo publish"" next. And double check if you did run ""git push --tags"""
}