#!/bin/sh
# Runs inside the WP-CLI container. Safe to run again: it skips what already exists.
# With NEW_APP_PASSWORD=1 it replaces the agent's application password and prints it
# as the last line; scripts/wp-up.sh sets it to 0 when .env already has a working one.
set -eu

if ! wp core is-installed 2>/dev/null; then
  wp core install --url="$WP_URL" --title="Agent demo" --admin_user="$WP_ADMIN_USER" \
    --admin_password="$WP_ADMIN_PASSWORD" --admin_email=admin@example.com --skip-email >&2
  wp rewrite structure '/%postname%/' --hard >&2
  wp post delete 1 --force >&2  # "Hello world!"

  for name in Backend DevOps Databases; do
    wp term create category "$name" >&2
  done
  for name in python django postgresql; do
    wp term create post_tag "$name" >&2
  done

  backend=$(wp term list category --name=Backend --field=term_id)
  wp post create --post_status=publish --post_category="$backend" --tags_input=django,python \
    --post_title="Django caching basics" \
    --post_content="<p>How the per-view cache, template fragment cache and low-level cache API fit together.</p>" >&2
  wp post create --post_status=publish --post_category="$backend" --tags_input=postgresql \
    --post_title="Reading a PostgreSQL query plan" \
    --post_content="<p>What EXPLAIN ANALYZE prints and which numbers matter first.</p>" >&2
fi

# Keep the site URL in step with WP_PORT, which may differ from the first install.
wp option update home "$WP_URL" >&2
wp option update siteurl "$WP_URL" >&2

[ "$NEW_APP_PASSWORD" = "1" ] || exit 0
for uuid in $(wp user application-password list "$WP_ADMIN_USER" --name=wpagent --field=uuid); do
  wp user application-password delete "$WP_ADMIN_USER" "$uuid" >&2
done
wp user application-password create "$WP_ADMIN_USER" wpagent --porcelain
