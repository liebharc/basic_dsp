$all_okay = $true

$last_master_update = $(git show --format="%ci" "origin/master" | Select -First 1)
$last_master_update = $(Get-Date $last_master_update)

# Release needs to be tagged
$current_version = $(Get-Content Cargo.toml | Select-String -Pattern "version\s+=\s+""(\d+\.\d+\.\d+)""" -AllMatches | % { $_.Matches.Groups[1].Value })
$current_tag = "v$current_version"
$tag_matches =  $(git tag | Where { $_ -eq "$current_tag" } | Measure).Count
if ($tag_matches -eq 0) {
    $Host.UI.WriteErrorLine("Tag $current_tag is missing")
    $all_okay = $false
}
else {
    $tag_date = $(git show --format="%ci" "$current_tag" | Select -Skip 4 -First 1);
    $tag_date = $(Get-Date $tag_date)
    $age_difference_days = ($last_master_update - $tag_date).TotalDays
    if ($age_difference_days -gt 3.0) {
        $Host.UI.WriteErrorLine("Tag $current_tag is $age_difference_days days older than the latest master commit, did you forget to update the lib version?")
        $all_okay = $false
    }
}

# Documentation needs to be up to date
$last_doc_update = $(git show --format="%ci" "origin/gh-pages" | Select -First 1)
$last_doc_update = $(Get-Date $last_doc_update)

$age_difference_days = ($last_master_update - $last_doc_update).TotalDays

if ($age_difference_days -gt 3.0) {
    $Host.UI.WriteErrorLine("Documentation hasn't been updated $age_difference_days days since the last master commit.")
    $all_okay = $false
}

if ($all_okay) {
    Write-Output "Everything seems to be okay, run ""cargo publish"" next"
}