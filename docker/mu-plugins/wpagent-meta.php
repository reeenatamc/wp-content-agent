<?php
/**
 * Plugin Name: WP Agent meta
 * Description: Exposes the meta_description post meta in the REST API.
 */

add_action('init', function () {
    register_post_meta('post', 'meta_description', [
        'show_in_rest' => true,
        'single' => true,
        'type' => 'string',
        'auth_callback' => fn() => current_user_can('edit_posts'),
    ]);
});
